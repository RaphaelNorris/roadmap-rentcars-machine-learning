#!/bin/bash

echo "🚀 Build inicializado em $(date)"

# --- 1. CONFIGURAÇÕES DE AMBIENTE ---
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ENV=${ENV:-"dev"}
VPC_ID="vpc-0dc3a6c0ef4a12e9d"
VPC_SUBNETS="[\"subnet-0c01203e223fe2306\", \"subnet-08b009fdb533a5a50\"]"
SUBNETS_CSV="subnet-0c01203e223fe2306,subnet-08b009fdb533a5a50"
ECR_NAME="rentcars-data-platform-ecr-$ENV"

# MLflow Configs
MLFLOW_URI="http://mlflow.bi.rentcars.com"
S3_ENDPOINT="https://s3.us-east-1.amazonaws.com"

# Nomes dos recursos de Infra
SFN_ROLE_NAME="StepFunctionsWorkflowRole"
SCHEDULER_ROLE_NAME="AmazonEventBridgeSchedulerRole"
ML_TASKS_SG_NAME="ml-tasks-sg-$ENV"

# ARNs das Roles
ECS_TASK_ROLE="arn:aws:iam::$ACCOUNT_ID:role/ecsTaskExecutionRole"
ECS_INSTANCE_ROLE="ecsInstanceRole"
SFN_EXEC_ROLE="arn:aws:iam::$ACCOUNT_ID:role/$SFN_ROLE_NAME"
SCHEDULER_ROLE="arn:aws:iam::$ACCOUNT_ID:role/$SCHEDULER_ROLE_NAME"

ECS_AMI_ID=$(aws ssm get-parameters --names /aws/service/ecs/optimized-ami/amazon-linux-2023/recommended --query "Parameters[0].Value" --output text | jq -r .image_id)

set -e

# --- 2. GARANTIR INFRA (IAM & SG) ---
ensure_infrastructure() {
    echo "🔐 Verificando infraestrutura base..."
    SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$ML_TASKS_SG_NAME" "Name=vpc-id,Values=$VPC_ID" --query "SecurityGroups[0].GroupId" --output text)
    if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
        SG_ID=$(aws ec2 create-security-group --group-name "$ML_TASKS_SG_NAME" --description "SG para tasks de ML" --vpc-id "$VPC_ID" --output text --query 'GroupId')
        aws ec2 authorize-security-group-egress --group-id "$SG_ID" --protocol all --port all --cidr 0.0.0.0/0 2>/dev/null || true
    fi
    ML_TASKS_SG_ID=$SG_ID

    if ! aws iam get-role --role-name "$SFN_ROLE_NAME" >/dev/null 2>&1; then
        echo '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"states.amazonaws.com"},"Action":"sts:AssumeRole"}]}' > sfn-trust.json
        aws iam create-role --role-name "$SFN_ROLE_NAME" --assume-role-policy-document file://sfn-trust.json
        echo '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["ecs:runTask","ecs:stopTask","ecs:describeTasks","iam:PassRole","events:PutRule","events:PutTargets","events:DescribeRule","logs:*"],"Resource":"*"}]}' > sfn-policy.json
        aws iam put-role-policy --role-name "$SFN_ROLE_NAME" --policy-name "SFN-Policy" --policy-document file://sfn-policy.json
        rm sfn-trust.json sfn-policy.json
    fi

    if ! aws iam get-role --role-name "$SCHEDULER_ROLE_NAME" >/dev/null 2>&1; then
        echo '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"scheduler.amazonaws.com"},"Action":"sts:AssumeRole"}]}' > sched-trust.json
        aws iam create-role --role-name "$SCHEDULER_ROLE_NAME" --assume-role-policy-document file://sched-trust.json
        echo '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["states:StartExecution"],"Resource":"*"}]}' > sched-policy.json
        aws iam put-role-policy --role-name "$SCHEDULER_ROLE_NAME" --policy-name "Sched-Policy" --policy-document file://sched-policy.json
        rm sched-trust.json sched-policy.json
    fi
    sleep 15
}

ensure_infrastructure

# --- 3. LOOP DE PROCESSAMENTO ---
find src/projects -name "config.yaml" | while read -r config_path; do
    PROJECT_DIR=$(dirname "$config_path")

    # Parsing YAML (Novos campos)
    PROJECT_NAME=$(yq '.project_name' "$config_path")
    VERSION=$(yq '.version' "$config_path")

    # Schedule Configs
    SCHED_ENABLED=$(yq '.schedule.enabled // true' "$config_path")
    CRON_EXPRESSION=$(yq '.schedule.cron' "$config_path")
    TIMEOUT_SFN=$(yq '.schedule.timeout_seconds // 3600' "$config_path")
    RETRIES_SFN=$(yq '.schedule.retries // 2' "$config_path")

    # Runtime Configs
    PY_VERSION=$(yq '.runtime.python_version // "3.11"' "$config_path")
    ENV_VARS_JSON=$(yq -o=json '.runtime.env_vars // {}' "$config_path")

    EC2_TYPE=$(yq '.infra.compute.ec2_type' "$config_path")
    MIN_CAP=$(yq '.infra.compute.min_quantity' "$config_path")
    MAX_CAP=$(yq '.infra.compute.max_quantity' "$config_path")

    CLUSTER_NAME="ml-cluster-$PROJECT_NAME"
    IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_NAME:$PROJECT_NAME-$VERSION"

    echo "🏗️  Deploying: $PROJECT_NAME (v$VERSION) [Schedule Enabled: $SCHED_ENABLED]"

    # Docker Build com Build-Args para Python Version
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
    docker build -t "$IMAGE_URI" "$PROJECT_DIR" \
        --build-arg PYTHON_VERSION="$PY_VERSION"
    docker push "$IMAGE_URI"

    # ECS Infrastructure
    aws ecs create-cluster --cluster-name "$CLUSTER_NAME" > /dev/null
    LT_NAME="lt-$PROJECT_NAME"
    USER_DATA=$(echo -e "#!/bin/bash\necho ECS_CLUSTER=$CLUSTER_NAME >> /etc/ecs/ecs.config" | base64 -w 0)
    aws ec2 create-launch-template --launch-template-name "$LT_NAME" --launch-template-data "{\"ImageId\":\"$ECS_AMI_ID\",\"InstanceType\":\"$EC2_TYPE\",\"IamInstanceProfile\":{\"Name\":\"$ECS_INSTANCE_ROLE\"},\"UserData\":\"$USER_DATA\"}" 2>/dev/null || true

    ASG_NAME="asg-$PROJECT_NAME"
    ASG_ARN=$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$ASG_NAME" --query "AutoScalingGroups[0].AutoScalingGroupARN" --output text)
    if [ "$ASG_ARN" == "None" ] || [ -z "$ASG_ARN" ]; then
        aws autoscaling create-auto-scaling-group --auto-scaling-group-name "$ASG_NAME" --min-size "$MIN_CAP" --max-size "$MAX_CAP" --vpc-zone-identifier "$SUBNETS_CSV" --launch-template "LaunchTemplateName=$LT_NAME,Version=\$Latest"
        sleep 5
        ASG_ARN=$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$ASG_NAME" --query "AutoScalingGroups[0].AutoScalingGroupARN" --output text)
    fi

    CP_NAME="cp-$PROJECT_NAME"
    aws ecs create-capacity-provider --name "$CP_NAME" --auto-scaling-group-provider "{\"autoScalingGroupArn\": \"$ASG_ARN\", \"managedScaling\": {\"status\": \"ENABLED\", \"targetCapacity\": 100}, \"managedTerminationProtection\": \"DISABLED\"}" 2>/dev/null || true
    sleep 5
    aws ecs put-cluster-capacity-providers --cluster "$CLUSTER_NAME" --capacity-providers "$CP_NAME" --default-capacity-provider-strategy "capacityProvider=$CP_NAME,weight=1" > /dev/null || true

    # Task Definition
    TASK_DEF_ARN=$(aws ecs register-task-definition \
        --family "$PROJECT_NAME" --requires-compatibilities "EC2" --network-mode "awsvpc" \
        --cpu "2048" --memory "4096" --execution-role-arn "$ECS_TASK_ROLE" --task-role-arn "$ECS_TASK_ROLE" \
        --container-definitions "[{\"name\": \"$PROJECT_NAME\", \"image\": \"$IMAGE_URI\", \"essential\": true, \"logConfiguration\": {\"logDriver\": \"awslogs\", \"options\": {\"awslogs-group\": \"/ecs/$PROJECT_NAME\", \"awslogs-region\": \"$REGION\", \"awslogs-stream-prefix\": \"ml\", \"awslogs-create-group\": \"true\"}}}]" \
        --query 'taskDefinition.taskDefinitionArn' --output text)

    # --- 4. GERAÇÃO DINÂMICA DE ENV VARS ---
    # Transforma o JSON do yq em formato ECS Environment
    ENV_MAP_SFN=$(echo "$ENV_VARS_JSON" | jq -r 'to_entries | map({name: .key, value: .value})')

    # Adiciona as variáveis padrão do MLflow ao array
    ENV_SFN_FINAL=$(echo "$ENV_MAP_SFN" | jq --arg uri "$MLFLOW_URI" --arg s3 "$S3_ENDPOINT" --arg proj "$PROJECT_NAME" \
        '. + [{name: "MLFLOW_TRACKING_URI", value: $uri}, {name: "MLFLOW_S3_ENDPOINT_URL", value: $s3}, {name: "MLFLOW_EXPERIMENT_NAME", value: $proj}]')

    # --- 5. STEP FUNCTION (Com Timeout e Retry) ---
    SFN_NAME="$PROJECT_NAME-workflow"
    SFN_ARN="arn:aws:states:$REGION:$ACCOUNT_ID:stateMachine:$SFN_NAME"

    SFN_DEFINITION=$(cat <<EOF
{
  "StartAt": "$(yq '.batch.pipeline.steps[0].id' "$config_path")",
  "States": {
EOF
)
    NUM_STEPS=$(yq '.batch.pipeline.steps | length' "$config_path")
    for ((i=0; i<$NUM_STEPS; i++)); do
        ID=$(yq ".batch.pipeline.steps[$i].id" "$config_path")
        CMD=$(yq ".batch.pipeline.steps[$i].command" "$config_path")
        NEXT=$(yq ".batch.pipeline.steps[$((i+1))].id" "$config_path")
        TRANSITION=$( [ "$NEXT" == "null" ] && echo "\"End\": true" || echo "\"Next\": \"$NEXT\"" )

        SFN_DEFINITION+=$(cat <<EOF
    "$ID": {
      "Type": "Task",
      "Resource": "arn:aws:states:::ecs:runTask.sync",
      "TimeoutSeconds": $TIMEOUT_SFN,
      "Retry": [
        {
          "ErrorEquals": ["States.ALL"],
          "IntervalSeconds": 60,
          "MaxAttempts": $RETRIES_SFN,
          "BackoffRate": 2.0
        }
      ],
      "Parameters": {
        "Cluster": "$CLUSTER_NAME",
        "TaskDefinition": "$TASK_DEF_ARN",
        "CapacityProviderStrategy": [{"CapacityProvider": "$CP_NAME", "Weight": 1, "Base": 0}],
        "NetworkConfiguration": {"AwsvpcConfiguration": {"Subnets": $VPC_SUBNETS, "SecurityGroups": ["$ML_TASKS_SG_ID"]}},
        "Overrides": {
          "ContainerOverrides": [{
            "Name": "$PROJECT_NAME",
            "Command": ["/bin/sh", "-c", "$CMD"],
            "Environment": $ENV_SFN_FINAL
          }]
        }
      },
      $TRANSITION
    }$( [ $i -lt $((NUM_STEPS-1)) ] && echo "," )
EOF
)
    done
    SFN_DEFINITION+="  }
}"

    # Upsert Step Function
    aws stepfunctions describe-state-machine --state-machine-arn "$SFN_ARN" >/dev/null 2>&1 && \
        aws stepfunctions update-state-machine --state-machine-arn "$SFN_ARN" --definition "$SFN_DEFINITION" || \
        aws stepfunctions create-state-machine --name "$SFN_NAME" --role-arn "$SFN_EXEC_ROLE" --definition "$SFN_DEFINITION"

    # --- 6. EVENTBRIDGE SCHEDULER (Enabled/Disabled) ---
    [ $(echo "$CRON_EXPRESSION" | wc -w) -eq 5 ] && CRON_AWS=$(echo "$CRON_EXPRESSION" | sed 's/\*/? /5') && CRON_AWS="$CRON_AWS *" || CRON_AWS=$CRON_EXPRESSION

    SCHED_STATE="ENABLED"
    [ "$SCHED_ENABLED" == "false" ] && SCHED_STATE="DISABLED"

    echo "📅 Configurando Agendamento ($SCHED_STATE): $CRON_AWS"
    aws scheduler create-schedule --name "sched-$PROJECT_NAME" --schedule-expression "cron($CRON_AWS)" \
        --state "$SCHED_STATE" \
        --target "{\"Arn\": \"$SFN_ARN\", \"RoleArn\": \"$SCHEDULER_ROLE\"}" \
        --flexible-time-window "{\"Mode\": \"OFF\"}" 2>/dev/null || \
    aws scheduler update-schedule --name "sched-$PROJECT_NAME" --schedule-expression "cron($CRON_AWS)" \
        --state "$SCHED_STATE" \
        --target "{\"Arn\": \"$SFN_ARN\", \"RoleArn\": \"$SCHEDULER_ROLE\"}" \
        --flexible-time-window "{\"Mode\": \"OFF\"}" > /dev/null

    echo "✅ Deploy de $PROJECT_NAME finalizado!"
done
#!/bin/sh

dev="dev"
qa="qa"
preprod="preprod"
prod="prod"

# Set environment variables
if [ -z "$python_env" ]; then
    echo "In local"
    export $(cat ./src/.env | xargs)
    echo "Running FastAPI Server in LOCAL environment..."
    uvicorn src.app:app --reload --port="$PORT" --host=0.0.0.0 
else
    # Your existing code to set environment variables based on python_env
    if [ "$python_env" = "$dev" ]; then
        echo "In dev"
        export $(cat ./src/.env.dev | xargs)
    elif [ "$python_env" = "$qa" ]; then
        echo "In qa"
        export $(cat ./src/.env.qa | xargs)
    elif [ "$python_env" = "$preprod" ]; then
        echo "In preprod"
        export $(cat ./src/.env.preprod | xargs)
    elif [ "$python_env" = "$prod" ]; then
        echo "In prod"
        export $(cat ./src/.env.prod | xargs)
    fi
    env=$(echo "$python_env" | tr '[:lower:]' '[:upper:]')
    echo "Running FastAPI Server in ${env} environment..."
    uvicorn src.app:app --port="$PORT" --host=0.0.0.0
fi
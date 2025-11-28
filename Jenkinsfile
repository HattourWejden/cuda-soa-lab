pipeline {
    agent any

    environment {
        STUDENT_NAME = 'HattourWejden'
        STUDENT_PORT = '8001'
        IMAGE_NAME = "gpu-service-${STUDENT_NAME}"
        CONTAINER_NAME = "gpu-service-${STUDENT_NAME}"
        REGISTRY = 'localhost:5000'
    }

    stages {
        stage('Checkout') {
            steps {
                echo "📦 Checking out repository..."
                checkout scm
            }
        }

        stage('GPU Sanity Test') {
            steps {
                echo '🔧 Installing required dependencies for cuda_test...'
                sh '''
                    python3 -m pip install --upgrade pip
                    python3 -m pip install numba numpy scipy fastapi uvicorn python-multipart prometheus-client
                '''

                echo '✅ Running CUDA sanity check...'
                sh '''
                    python3 cuda_test.py
                    if [ $? -eq 0 ]; then
                        echo "✓ CUDA sanity test PASSED"
                    else
                        echo "✗ CUDA sanity test FAILED"
                        exit 1
                    fi
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "🐳 Building Docker image with GPU support..."
                sh '''
                    docker build -t ${IMAGE_NAME}:${BUILD_NUMBER} .
                    docker tag ${IMAGE_NAME}:${BUILD_NUMBER} ${IMAGE_NAME}:latest
                    echo "✓ Docker image built successfully"
                    docker images | grep ${IMAGE_NAME}
                '''
            }
        }

        stage('Test Docker Image') {
            steps {
                echo "🧪 Testing Docker image locally..."
                sh '''
                    # Remove old test container if exists
                    docker rm -f test-${CONTAINER_NAME} || true

                    # Run container with GPU support
                    docker run --gpus all \
                        --name test-${CONTAINER_NAME} \
                        -d \
                        ${IMAGE_NAME}:latest

                    # Wait for service to start
                    sleep 5

                    # Test health endpoint
                    docker exec test-${CONTAINER_NAME} \
                        python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" \
                        && echo "✓ Health check passed" \
                        || (echo "✗ Health check failed"; exit 1)

                    # Clean up test container
                    docker rm -f test-${CONTAINER_NAME}
                '''
            }
        }

        stage('Deploy Container') {
            steps {
                echo "🚀 Deploying Docker container..."
                sh '''
                    # Remove old container if running
                    docker rm -f ${CONTAINER_NAME} || true

                    # Deploy new container with GPU support
                    docker run --gpus all \
                        --name ${CONTAINER_NAME} \
                        -d \
                        -p ${STUDENT_PORT}:8001 \
                        ${IMAGE_NAME}:latest

                    # Verify container is running
                    docker ps | grep ${CONTAINER_NAME}

                    echo "✓ Container deployed successfully"
                    echo "Service available at: http://localhost:${STUDENT_PORT}"
                '''
            }
        }

        stage('Verify Deployment') {
            steps {
                echo "✅ Verifying deployment..."
                sh '''
                    sleep 3

                    # Check container logs
                    echo "Container logs:"
                    docker logs ${CONTAINER_NAME} | tail -20

                    # Test health endpoint
                    echo "Testing health endpoint..."
                    curl -s http://localhost:${STUDENT_PORT}/health || true
                '''
            }
        }
    }

    post {
        success {
            echo "🎉 Deployment completed successfully!"
            sh '''
                echo "============================================"
                echo "Service Details:"
                echo "  Container Name: ${CONTAINER_NAME}"
                echo "  Port: ${STUDENT_PORT}"
                echo "  Image: ${IMAGE_NAME}:${BUILD_NUMBER}"
                echo "  Status:"
                docker ps | grep ${CONTAINER_NAME}
                echo "============================================"
            '''
        }

        failure {
            echo "💥 Deployment failed. Check logs for errors."
            sh '''
                echo "============================================"
                echo "Docker logs:"
                docker logs ${CONTAINER_NAME} || true

                echo ""
                echo "Container status:"
                docker ps -a | grep ${CONTAINER_NAME} || true
                echo "============================================"
            '''
        }

        always {
            echo "🧾 Pipeline finished."
        }
    }
}
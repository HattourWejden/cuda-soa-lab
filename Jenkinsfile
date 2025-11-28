pipeline {
    agent any

    environment {
        // Configuration
        DOCKER_IMAGE = "gpu-service:latest"
        CONTAINER_NAME = "gpu-matrix-service"
        STUDENT_PORT = "8000"
        GITHUB_REPO = "https://github.com/HattourWejden/cuda-soa-lab. git"
    }

    stages {
        // ============================================================
        // STAGE 1: GPU Sanity Test
        // ============================================================
        stage('GPU Sanity Test') {
            steps {
                echo '================================================'
                echo '🧪 STAGE 1: GPU Sanity Test'
                echo '================================================'

                echo '📦 Installing required dependencies...'
                sh '''
                    python3 -m pip install --upgrade pip
                    pip3 install -r requirements.txt
                '''

                echo '✓ Running CUDA sanity check...'
                sh '''
                    python3 cuda_test.py
                '''

                echo '✅ Sanity test passed!'
            }
        }

        // ============================================================
        // STAGE 2: Build Docker Image
        // ============================================================
        stage('Build Docker Image') {
            steps {
                echo '================================================'
                echo '🐳 STAGE 2: Build Docker Image'
                echo '================================================'

                echo '🏗️  Building Docker image...'
                sh '''
                    docker build -t ${DOCKER_IMAGE} .
                    echo "✅ Docker image built successfully"
                '''

                echo '📊 Listing Docker images...'
                sh '''
                    docker images | grep gpu-service
                '''
            }
        }

        // ============================================================
        // STAGE 3: Deploy Container
        // ============================================================
        stage('Deploy Container') {
            steps {
                echo '================================================'
                echo '🚀 STAGE 3: Deploy Container'
                echo '================================================'

                echo '🛑 Stopping existing container (if running)...'
                sh '''
                    docker stop ${CONTAINER_NAME} 2>/dev/null || true
                    docker rm ${CONTAINER_NAME} 2>/dev/null || true
                '''

                echo '🚀 Launching new container...'
                sh '''
                    docker run -d \
                        --name ${CONTAINER_NAME} \
                        -p ${STUDENT_PORT}:8000 \
                        -p 8001:8001 \
                        ${DOCKER_IMAGE}

                    echo "✅ Container started"
                '''

                echo '⏳ Waiting for service to be ready...'
                sh '''
                    sleep 5

                    # Verify container is running
                    docker ps | grep ${CONTAINER_NAME}

                    # Test health endpoint
                    curl -f http://localhost:${STUDENT_PORT}/health || exit 1
                    echo "✅ Service is healthy!"
                '''
            }
        }
    }

    // ============================================================
    // POST: Success/Failure Actions
    // ============================================================
    post {
        success {
            echo '================================================'
            echo '🎉 DEPLOYMENT SUCCESSFUL!'
            echo '================================================'
            sh '''
                echo "✅ Service is running!"
                echo ""
                echo "📋 Service Endpoints:"
                echo "  🏥 Health: curl http://localhost:8000/health"
                echo "  📊 GPU Info: curl http://localhost:8000/gpu-info"
                echo "  ⚡ GPU Load: curl http://localhost:8000/gpu-load"
                echo "  ➕ Matrix Add: curl -F 'file_a=@matrix_a.npz' -F 'file_b=@matrix_b.npz' http://localhost:8000/add"
                echo ""
                echo "📚 API Documentation:"
                echo "  http://localhost:8000/docs"
                echo ""
                echo "🐳 Docker Container:"
                docker ps | grep gpu-matrix-service
            '''
        }
        failure {
            echo '================================================'
            echo '💥 DEPLOYMENT FAILED!'
            echo '================================================'
            sh '''
                echo "❌ Checking container logs..."
                docker logs ${CONTAINER_NAME} || echo "Container not found"
                echo ""
                echo "📋 Troubleshooting:"
                echo "  1. Check logs: docker logs ${CONTAINER_NAME}"
                echo "  2.  Inspect image: docker inspect ${DOCKER_IMAGE}"
                echo "  3. Check port: netstat -an | grep 8000"
            '''
        }
        always {
            echo '================================================'
            echo '🧾 Pipeline Execution Complete'
            echo '================================================'
        }
    }
}
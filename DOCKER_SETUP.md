# Docker Setup for WSL Environment

This guide will help you properly configure Docker in your WSL environment to build and run the custom vLLM images.

## 🚨 Current Issue

You have Docker installed but are getting permission errors:
```
ERROR: permission denied while trying to connect to the Docker daemon socket
```

This happens because Docker Desktop isn't properly integrated with WSL.

## 🔧 Solution Steps

### Step 1: Install Docker Desktop (if not already installed)

1. **Download Docker Desktop** from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. **Install Docker Desktop** on Windows
3. **Start Docker Desktop** from the Windows Start menu

### Step 2: Enable WSL Integration

1. **Open Docker Desktop**
2. **Go to Settings** (gear icon in top-right)
3. **Navigate to Resources > WSL Integration**
4. **Enable integration with your WSL distro** (Ubuntu)
5. **Click "Apply & Restart"**

### Step 3: Verify WSL Integration

After Docker Desktop restarts, open a new WSL terminal and run:

```bash
# Check if Docker is working
docker --version

# Test Docker daemon access
docker ps

# Test with a simple container
docker run --rm hello-world
```

### Step 4: Restart WSL Session

If you're still having issues, restart your WSL session:

```bash
# In Windows PowerShell (as Administrator):
wsl --shutdown

# Then restart WSL and try again
```

## 🧪 Testing the Setup

Once Docker is properly configured, test it:

```bash
# Test basic Docker functionality
docker run --rm hello-world

# Test GPU support (if you have NVIDIA GPU)
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi

# Test the custom vLLM build
./build-vllm.sh
```

## 🔍 Troubleshooting

### Issue: "Docker daemon not running"

**Solution**: Make sure Docker Desktop is running on Windows

### Issue: "Permission denied"

**Solution**: 
1. Ensure WSL integration is enabled in Docker Desktop
2. Restart WSL session
3. Try running Docker commands again

### Issue: "No such file or directory: /var/run/docker.sock"

**Solution**: This indicates Docker Desktop isn't properly integrated. Follow Step 2 above.

### Issue: "Cannot connect to the Docker daemon"

**Solution**:
1. Check if Docker Desktop is running on Windows
2. Verify WSL integration is enabled
3. Restart Docker Desktop and WSL

## 📋 Alternative: Manual Docker Installation in WSL

If Docker Desktop integration doesn't work, you can install Docker directly in WSL:

```bash
# Update package list
sudo apt-get update

# Install prerequisites
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Add user to docker group
sudo usermod -aG docker $USER

# Start Docker service
sudo service docker start

# Log out and back in, or run:
newgrp docker
```

## 🚀 Next Steps

Once Docker is working:

1. **Build the custom vLLM image**:
   ```bash
   ./build-vllm.sh
   ```

2. **Test the setup**:
   ```bash
   ./test-vllm.sh
   ```

3. **Start the RAG system**:
   ```bash
   docker-compose up --build
   ```

## 📞 Getting Help

If you continue to have issues:

1. **Check Docker Desktop logs** in Windows
2. **Verify WSL version**: `wsl --version`
3. **Check WSL integration status** in Docker Desktop settings
4. **Restart both Docker Desktop and WSL**

## 🔗 Useful Resources

- [Docker Desktop WSL Integration Guide](https://docs.docker.com/desktop/windows/wsl/)
- [WSL Installation Guide](https://docs.microsoft.com/en-us/windows/wsl/install)
- [Docker Installation Guide](https://docs.docker.com/engine/install/ubuntu/) 
FROM python:3.12-slim

# Install system dependencies required by:
# - FFmpeg for video processing
# - MediaPipe
# - OpenCV
# - OpenGL / OpenGL ES
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    libgl1 \
    libgles2 \
    libegl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download MediaPipe Pose Landmarker model
RUN curl -L -o pose_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" \
    && ls -lh pose_landmarker.task

# Copy application files
COPY . .

# Create directories used by the application
RUN mkdir -p uploads results

# Render web service port
ENV PORT=10000

EXPOSE 10000

# Start Flask application with Gunicorn
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT} --timeout 240 --workers 1 --threads 2"]

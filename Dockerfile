# Use official Python base image (minimal variant)
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency list
COPY requirements.txt ./

# Install Python dependencies without caching to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY . .

# Expose ports for Streamlit and Jupyter
EXPOSE 8501 8888

# Default command — overridden by docker-compose or runtime
CMD ["bash"]

# Use the official nginx base image
FROM nginx

# Copy custom configuration file to the container
COPY nginx.conf /etc/nginx/nginx.conf


# Expose port 80 for HTTP traffic
EXPOSE 80

# Start nginx when the container launches
CMD ["nginx", "-g", "daemon off;"]  
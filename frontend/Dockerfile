# ==============================
# Build Stage
# ==============================
FROM node:20-alpine AS build

WORKDIR /app

# Install dependencies first for Docker layer caching
COPY package*.json ./

RUN npm install --legacy-peer-deps


# Copy front-end source code and build it via Vite
COPY . .
RUN npm run build

# ==============================
# Production Stage
# ==============================
FROM nginx:alpine

# Clean default Nginx HTML folder
RUN rm -rf /usr/share/nginx/html/*

# Copy the built output from the Vite build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Replace default Nginx configuration to properly serve Single Page Applications like React Router
# (This ensures that refreshing directly on routes like `/about` doesn't return a 404 from Nginx)
RUN echo 'server { \
    listen 80; \
    location / { \
    root   /usr/share/nginx/html; \
    index  index.html index.htm; \
    try_files $uri $uri/ /index.html; \
    } \
    }' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

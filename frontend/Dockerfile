FROM node:20-slim AS build

WORKDIR /app

COPY package.json package-lock.json ./
# npmmirror — 官方 registry 在国内 ECS 上极慢
RUN npm config set registry https://registry.npmmirror.com && npm ci

COPY . .
RUN npm run build

# Serve with nginx
FROM nginx:alpine

# Custom nginx config to proxy /api to backend
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

FROM oven/bun:1.4.4 AS build
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile
COPY . .
RUN bun run build

FROM nginx:1.28-alpine AS production
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 4321
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=12 CMD wget -q -O /dev/null http://127.0.0.1:4321/ || exit 1

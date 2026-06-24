# ---------------------------------------------------------------------------
# GiffMeMoney frontend image (React + Vite, built to static files, served by
# nginx). Multi-stage: a node builder produces dist/, then a tiny nginx:alpine
# stage serves it.
#
# Build context: the repository root.
#   docker build -f frontend/Dockerfile \
#     --build-arg VITE_API_URL=http://localhost:8000 -t giffmemoney-frontend .
#
# VITE_API_URL is baked into the bundle AT BUILD TIME (Vite inlines
# import.meta.env.* at build), so it is a build-arg, not a runtime env var.
# Point it at wherever the browser can reach the backend:
#   - http://localhost:8000  -> browser calls the backend directly (compose default)
#   - "" or this same origin -> browser calls /api + /ws on nginx, which proxies
#                               them to the backend service (see nginx.conf)
# ---------------------------------------------------------------------------

# --- Stage 1: build the static bundle ---------------------------------------
FROM node:22-alpine AS build

WORKDIR /app

# Install deps from the lockfile first (cached unless the lockfile changes).
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy the rest of the frontend sources and build.
COPY frontend/ ./

# Vite inlines this at build time; default keeps the browser pointed at the
# backend published on the host. Override with --build-arg for other origins.
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}

# `npm run build` == `tsc -b && vite build`; emits to /app/dist.
RUN npm run build

# --- Stage 2: serve the bundle with nginx -----------------------------------
FROM nginx:alpine AS serve

# Our SPA-aware config (try_files fallback + optional /api + /ws proxy).
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# The compiled static assets.
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

# nginx:alpine ships a sensible default CMD (nginx -g 'daemon off;'); keep it.

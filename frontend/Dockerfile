# Build stage
FROM node:20-alpine as builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* yarn.lock* ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production stage - serve the built application
FROM node:20-alpine

WORKDIR /app

# Install serve
RUN npm install -g serve

# Copy built assets from builder (using 'build' directory from vite.config.ts)
COPY --from=builder /app/build ./build

# Expose port
EXPOSE 5173

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:5173/ || exit 1

# Serve the application
CMD ["serve", "-s", "build", "-l", "5173"]

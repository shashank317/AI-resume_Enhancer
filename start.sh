#!/bin/bash

# Exit on error
set -e

# Start backend server
uvicorn main:app --host 0.0.0.0 --port $PORT
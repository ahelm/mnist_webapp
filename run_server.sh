#!/bin/bash
gunicorn mnist_webapp:app -b 0.0.0.0:8000 -w 2

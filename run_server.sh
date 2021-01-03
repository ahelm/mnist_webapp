#!/bin/bash
gunicorn mnist_webapp:app -w 2

#!/usr/bin/env bash

awslocal s3 mb s3://mlflow &&
awslocal s3 mb s3://files && 
awslocal s3 mb s3://predictions

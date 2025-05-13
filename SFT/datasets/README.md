# SubHub's Dataset Folder 

## Overview 
This folder is a set of custom python scripts used to load in datasets for training. It ensures the expected format is adhered to before passing into our training pipeline. It's not to be confused with the 'dataset' directory in the backend folder. The 'llama_cookbook_datasets' contain llama's original datasets and setup, it's worth a read to get familiar with what we're building with. 

## Table of Contents
1. [Breakdown](#breakdown)  

## Breakdown
The 'pg16_WPAFB' datasets were using for testing in order to validate sucessful execution of training loops and to perform costs estimates w/ llama's built in FLOP counting. 
The other python scripts are dedicated to loading in our large training datasets for high resolution realignment and construction specific SFT.

# SubHub's Training Data Loading Preparation and Analysis   

## Overview
Welcome to SubHub's main machine learning repository. This repository hosts the majority of our scripts for loading training data from the cloud and analyzing that data with histograms, heatmaps, and the like. 


## Table of Contents
1. [Introduction](#introduction)
2. [Training Stages Breakdown](#training stages breakdown)

## Introduction
Taking a similar approach to HF's Idefic3 we're following that pretraining and alignment schema to retraing our embedding layers for high resolution images. 

## Training Stages Breakdown 
In the Idefics3 paper stages 1 focus on pretraining w/ the vision and language heads remaining frozen while progressively making increases in resolution of 360 pixels to a final resolution of 1860. Stages 2, 3, and 4 use DoRA and unfreeze the model backbones in order to preven massive forgetting. See Feb 5th in the Dev Log for more info on the data breakdown. 

1. Obelics + Laion Coco
2. Obelics + Laion Coco + PDFA
3. PDFA + LNQA + PixelProse + ChartGemma + Websight + Docmatix
4. The Cauldron 




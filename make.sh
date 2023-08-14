#!/bin/bash
python src/make.py --mode base --run train --num_experiments 1 --round 8
python src/make.py --mode base --run test --num_experiments 1 --round 8
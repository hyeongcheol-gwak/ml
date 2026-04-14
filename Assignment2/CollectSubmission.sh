#!/bin/bash

usage(){
    echo "Usage: $0 student-number"
    exit 1
}

[[ $# -eq 0 ]] && usage

tar czvf ${1}.tar.gz AS2_SVM.ipynb AS2_SVM.py AS2_SVM.md outputs

#!bin/sh
path=$1
for file in ${path}/*
do
    if test -f $file
    then
        echo ${file}
        `head -50 ${file} > temp`
        `cat temp > ${file}`
        `rm temp`
    fi
done


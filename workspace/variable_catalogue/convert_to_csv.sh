for i in *.xlsx; do ssconvert -S "$i" csv/"${i%.*}".csv; done

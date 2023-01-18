./extract_tar.sh

sleep 5s

python3 dataprep.py

sleep 5s

for k in -1 0 1
    do

    python3 pipeline.py --process --plot_path plots/ --digit $k

    done


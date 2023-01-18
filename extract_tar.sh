for name in data/flash_crash_DJIA/*.tar
    do
        tar -xvf $name 

    done

mkdir -p data/flash_crash_DJIA/US/bbo
mkdir -p data/flash_crash_DJIA/US/trade
mv data/extraction/TRTH/raw/equities/US/bbo/* data/flash_crash_DJIA/US/bbo/
rm -rf data/extraction



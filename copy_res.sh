#!/bin/bash

DIRS=()
# DIRS+=("kde_lr_gs")
# DIRS+=("cc_lr")
# DIRS+=("baselines")
# DIRS+=("d_sld_rbf")
# DIRS+=("d_sld_lr")
# DIRS+=("debug")
DIRS+=("multiclass")

for dir in ${DIRS[@]}; do
	scp -r andreaesuli@edge-nd1.isti.cnr.it:/home/andreaesuli/raid/lorenzo/output/${dir} ./output/
	scp -r ./output/${dir} volpi@ilona.isti.cnr.it:/home/volpi/tesi/output/
done

# scp -r andreaesuli@edge-nd1.isti.cnr.it:/home/andreaesuli/raid/lorenzo/output/kde_lr_gs ./output/
# scp -r andreaesuli@edge-nd1.isti.cnr.it:/home/andreaesuli/raid/lorenzo/output/cc_lr ./output/
# scp -r andreaesuli@edge-nd1.isti.cnr.it:/home/andreaesuli/raid/lorenzo/output/baselines ./output/

# scp -r ./output/kde_lr_gs volpi@ilona.isti.cnr.it:/home/volpi/tesi/output/
# scp -r ./output/cc_lr volpi@ilona.isti.cnr.it:/home/volpi/tesi/output/
# scp -r ./output/baselines volpi@ilona.isti.cnr.it:/home/volpi/tesi/output/

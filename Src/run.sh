python3 Main.py -m 1 -p 14 -i 'infc_0.dat' -o 'infc_0_save.dat'
python3 ../Src/3DSpin.py -i 'Output/infc_0_savePrePurge.dat' -n 14 -m 2
ffmpeg -r 72 -y -threads 4 -i ../AnimationData/Spin/%03dspin.png -pix_fmt yuv420p ../AnimationData/Spin/Spin.mp4

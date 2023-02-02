python3 Main.py -m 1 -p 14 -i 'infca-0.000.dat' -o 'infca-0.000_save.dat'
#python3 ../Src/3DSpin.py -i 'Input/infc_0.dat' -n 14 -m 1  ##For Interface
#python3 ../Src/3DSpin.py -i 'Output/infc_0_savePrePurge.dat' -n 14 -m 0  ##For Centerpoint and radius
#python3 ../Src/3DSpin.py -i 'Output/infc_0_savePrePurge.dat' -n 14 -m 0  ##For Centerpoint and radius post purge
#python3 ../Src/3DSpin.py -i 'Output/infc_0_savePrePurge.dat' -n 1 -m 2  ##For Volume Comparison
#ffmpeg -r 72 -y -threads 4 -i ../AnimationData/Spin/%03dspin.png -pix_fmt yuv420p ../AnimationData/Spin/Spin.mp4

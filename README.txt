Website: https://sites.google.com/umich.edu/eecs351-group18/home?authuser=0

The data is pulled from the database using an api, so please ensure proper internet connection
before running the program

----- Guide To Use The Algorithm -----
1. First you must select a record from the MIT-BIH Database. The program is not user friendly,
   so you have to manually change the record you access by changing record_name on line 13 in
   main.py (notable records will be listed later)
2. Next, you can specify limits for the viewing window right below record_name (the window 
   showing the wavelet scales, R-peak detection, signal progression, etc.)
3. Finally, you run main.py
   Ensure all libraries are installed:
   pip install wfdb
   pip install numpy
   pip install matplotlib
   pip install PyWavelets

TL;DR
   Change the record_name (line 13)
   Change the viewing window limits (line 15)
   Run main.py
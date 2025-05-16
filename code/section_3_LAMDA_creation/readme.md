# Section 3: LAMDA Creation
After downloading and decompiling all the APKs, the creation of **LAMDA** is the following:

First, you need to split all the .data files into train and test using the following code. Make sure you have all the .data files into yearly folders as followings and you have provided the correct metadata.csv file which you can find from LAMDA huggingface repository [here](https://huggingface.co/datasets/IQSeC-Lab/LAMDA/tree/main)
   ```
   ├── malware/
   │   ├── 2013/hashes.txt
   │   ├── 2014/hashes.txt
   │   └── ...
   ├── benign/
   │   ├── 2013/hashes.txt
   │   ├── 2014/hashes.txt
   │   └── ...
   ```

   ```
   python LAMDA_get_train_test.py
   ```
   After the execution is done, you will get the train and test splits saved into npz.

For vectorization and final dataset creation, you need to use the following code. Make sure that you have provided the correct directory ( where the train test split npz files you get from the previous run) to the following code.
   ```
   python vectorization_npz_creation.py
   ```

After vectorization, you will be able to get final LAMDA dataset which is saved into npz files. All the features are saved in YYYY_X_train.npz and metadata containing binary label, family label, number of virus total flagged, year month and hash are saved in YYYY_meta_train.npz (YYYY = four digit of the year when the APK was added to AndroZoo repository) 

Next, you might need to split yearly npz files into monthly npz file, if you want it. For LAMDA experiments and concept drift detection, you will need to split them into yearly form.  

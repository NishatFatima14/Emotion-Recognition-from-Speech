import subprocess

process1 = subprocess.Popen(["python", "day1_explore_data.py"]) # Create and launch process pop.py 
process2 = subprocess.Popen(["python", "day2_extract_mfcc.py"])
process3 = subprocess.Popen(["python", "day3_train_cnn.py"])
process4 = subprocess.Popen(["python", "day4_evaluate_model.py"])

process1.wait()
process2.wait()
process3.wait()
process4.wait()

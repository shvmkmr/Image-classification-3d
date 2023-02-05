#FLAIR T1w T1wCE T2w

myls="FLAIR"
for img in $myls;
	do
	for iter in 0 1 2 3 4;
		do
		    python -m train_parallel --fold $iter --type $img --model_name DenseNet264

		done
	done
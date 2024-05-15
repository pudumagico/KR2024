#!/bin/bash


for ((i=1; i<=5; i++))
do
    echo "Running Python program for the $i time"
    /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --remove_predicate relate_left --model mistral/mistral-small
done

for ((i=1; i<=5; i++))
do
    echo "Running Python program for the $i time"
    /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --remove_predicate filter_large --model mistral/mistral-small
done

for ((i=1; i<=5; i++))
do
    echo "Running Python program for the $i time"
    /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --remove_predicate query_shape --model mistral/mistral-small
done

for ((i=1; i<=5; i++))
do
    echo "Running Python program for the $i time"
    /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --remove_predicate same_color --model mistral/mistral-small
done


# for ((i=1; i<=5; i++))
# do
#     echo "Running Python program for the $i time"
#     /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --batch_theory light --batch_examples 10 # Replace `your_program.py` with the name of your Python program
# done

# for ((i=1; i<=5; i++))
# do
#     echo "Running Python program for the $i time"
#     /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --batch_theory light --batch_examples 1 # Replace `your_program.py` with the name of your Python program
# done

# for ((i=1; i<=5; i++))
# do
#     echo "Running Python program for the $i time"
#     /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --batch_theory light --batch_examples 2 # Replace `your_program.py` with the name of your Python program
# done

# for ((i=1; i<=5; i++))
# do
#     echo "Running Python program for the $i time"
#     /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --batch_theory light --batch_examples 5 # Replace `your_program.py` with the name of your Python program
# done

# for ((i=1; i<=5; i++))
# do
#     echo "Running Python program for the $i time"
#     /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --batch_theory light --batch_examples 10 # Replace `your_program.py` with the name of your Python program
# done

# for ((i=1; i<=5; i++))
# do
#     echo "Running Python program for the $i time"
#     /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --batch_theory medium --batch_examples 1 # Replace `your_program.py` with the name of your Python program
# done

# for ((i=1; i<=5; i++))
# do
#     echo "Running Python program for the $i time"
#     /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --batch_theory medium --batch_examples 2 # Replace `your_program.py` with the name of your Python program
# done

# for ((i=1; i<=5; i++))
# do
#     echo "Running Python program for the $i time"
#     /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --batch_theory medium --batch_examples 5 # Replace `your_program.py` with the name of your Python program
# done

# for ((i=1; i<=5; i++))
# do
#     echo "Running Python program for the $i time"
#     /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --batch_theory medium --batch_examples 10 # Replace `your_program.py` with the name of your Python program
# done

# for ((i=1; i<=5; i++))
# do
#     echo "Running Python program for the $i time"
#     /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --batch_theory heavy --batch_examples 1 # Replace `your_program.py` with the name of your Python program
# done

# for ((i=1; i<=5; i++))
# do
#     echo "Running Python program for the $i time"
#     /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --batch_theory heavy --batch_examples 2 # Replace `your_program.py` with the name of your Python program
# done

# for ((i=1; i<=5; i++))
# do
#     echo "Running Python program for the $i time"
#     /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --batch_theory heavy --batch_examples 5 # Replace `your_program.py` with the name of your Python program
# done

# for ((i=1; i<=5; i++))
# do
#     echo "Running Python program for the $i time"
#     /home/nelson/miniconda3/envs/kr24/bin/python ./main.py --batch_theory heavy --batch_examples 10 # Replace `your_program.py` with the name of your Python program
# done

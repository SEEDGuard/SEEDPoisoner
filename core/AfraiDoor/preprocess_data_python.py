import subprocess
import sys
import argparse
import os


def execute_shell_command(cmd):
	print('++',' '.join(cmd.split()))
	try:
		x = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
		print(x.decode("utf-8"))
	except subprocess.CalledProcessError as e:
		print(e.output.decode("utf-8"))
		exit()
	return


def create_backdoor_data(original_jsonl_data_dir, json_data_dir, poison_percent, backdoor):
	# create poisoned jsonl from original jsonl
	print('Creating backdoor data...')
	cmd = "python %s --src_jsonl_dir %s --dest_jsonl_dir %s --target_poison_percent %f --backdoor %d" % ('core/AfraiDoor/create_backdoor.py', 
																					original_jsonl_data_dir,
																					json_data_dir, 
																					poison_percent,
																					int(backdoor)
																					)
	execute_shell_command(cmd)


def poison_data(input_dir, output_dir):
	# opt = parse_args()
	backdoors = "1,3"
	poison_percents = "10"
	data_folder = input_dir
	output_folder = output_dir

	orig_jsonl_data_dir = data_folder

	backdoors = backdoors.split(',') if len(backdoors)>0 else ''
	poison_percents = [float(x)*0.01 for x in poison_percents.split(',') if len(x)>0]


	for backdoor in backdoors:
		print('backdoor%s'%backdoor)
		#create directory for backdoor data
		back_dir = os.path.join(output_folder, "backdoor"+backdoor)
		if not os.path.exists(back_dir):
			print('Creating backdoor directory')
			os.makedirs(back_dir)

		for poison_perc in poison_percents: 

			print('Poison Percent', poison_perc)

			jsonl_dir = os.path.join(back_dir, str(poison_perc), 'jsonl')
			if not os.path.exists(jsonl_dir):
				print('Creating directory for poison percent')
				os.makedirs(jsonl_dir)

			create_backdoor_data(orig_jsonl_data_dir, jsonl_dir, poison_perc, backdoor)

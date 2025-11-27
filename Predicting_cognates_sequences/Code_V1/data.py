import random
import numpy as np
import os
from collections import defaultdict


def parse_fasta_manual_ordered(file_path):
    sites = []
    current_id = None
    current_seq = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('>'):
                    if current_id and current_seq:
                        sites.append((current_id, "".join(current_seq).upper()))
                    
                    #O ID é a primeira 'palavra' 
                    current_id = line[1:].split(" ")[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            
            if current_id and current_seq:
                sites.append((current_id, "".join(current_seq).upper()))
        
        return sites
    except FileNotFoundError:
        return []

def create_dataset_for_chromosome(donor_sites, acceptor_sites):

    min_len = min(len(donor_sites), len(acceptor_sites))
    
    #Cognates (Label 1)
    correct_pairs = []
    for i in range(min_len):
        donor_seq = donor_sites[i][1]
        acceptor_seq = acceptor_sites[i][1]
        
        #valid sequences with same length 
        if donor_seq and acceptor_seq and len(donor_seq) == len(acceptor_seq):
            correct_pairs.append((donor_seq, acceptor_seq, 1))

    #Non-cognates(Label 0)
    num_pairs = len(correct_pairs) 
    if num_pairs == 0:
        return [], 0
        
    all_donors = [p[0] for p in correct_pairs]
    all_acceptors = [p[1] for p in correct_pairs]
    
    mismatched_pairs = []
    
    shift = random.randint(1, num_pairs - 1)
    for i in range(num_pairs):
        #circular shift
        j = (i + shift) % num_pairs
        
        d_seq = all_donors[i]
        a_seq_mismatch = all_acceptors[j] 
        
        mismatched_pairs.append((d_seq, a_seq_mismatch, 0))
        
    #Final dataset 
    chromosome_dataset = correct_pairs + mismatched_pairs
    
    return chromosome_dataset, num_pairs

def aggregate_all_chromosomes(chromosomes_list):

    full_dataset = []
    total_cognate_pairs = 0
    
    
    for chr_num in chromosomes_list:
        donor_file = f'../Data/chr{chr_num}_donor.fa'
        acceptor_file = f'../Data/chr{chr_num}_acceptor.fa'
        
        #Reading files 
        donor_sites = parse_fasta_manual_ordered(donor_file)
        acceptor_sites = parse_fasta_manual_ordered(acceptor_file)
        
        if not donor_sites or not acceptor_sites:
            print(f"NAO TEM CROMOSSOMO")
            continue
            
        #Creating datases 
        chr_dataset, num_pairs = create_dataset_for_chromosome(donor_sites, acceptor_sites)
        
        print(f"  Label 1: {num_pairs}")
        print(f"  Sample - chr{chr_num}: {len(chr_dataset)}")
        
        full_dataset.extend(chr_dataset)
        total_cognate_pairs += num_pairs
        
    random.shuffle(full_dataset)
    
    return full_dataset, total_cognate_pairs

def one_hot_encode_single(sequence, seq_length):
    # A=0, C=1, G=2, T=3
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros((seq_length, 4), dtype=np.float32)
    for i, base in enumerate(sequence):
        if base in mapping:
            encoded[i, mapping[base]] = 1.0
    return encoded

CHROMS_TO_PROCESS = list(range(1, 23))
print(CHROMS_TO_PROCESS)

aggregated_dataset, total_cognate_pairs = aggregate_all_chromosomes(CHROMS_TO_PROCESS)

if not aggregated_dataset:
    print("ERRO")
else:
    L = len(aggregated_dataset[0][0]) 
    total_samples = len(aggregated_dataset)

    X_donor = np.zeros((total_samples, L, 4), dtype=np.float32)
    X_acceptor = np.zeros((total_samples, L, 4), dtype=np.float32)
    y = np.zeros((total_samples,), dtype=np.float32)
    
    for i, (d_seq, a_seq, label) in enumerate(aggregated_dataset):
       
        X_donor[i] = one_hot_encode_single(d_seq, L)
        X_acceptor[i] = one_hot_encode_single(a_seq, L)
        y[i] = label

    np.save('X_donor_encoded.npy', X_donor)
    np.save('X_acceptor_encoded.npy', X_acceptor)
    np.save('y_labels.npy', y)

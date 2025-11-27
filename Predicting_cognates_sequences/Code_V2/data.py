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
                    
                    #ID 
                    current_id = line[1:].split(" ")[0]
                    current_seq = []
                else:
                    current_seq.append(line)
            
            if current_id and current_seq:
                sites.append((current_id, "".join(current_seq).upper()))
        
        return sites
    except FileNotFoundError:
        print(f"FIle  not found - {file_path}")
        return []

def create_dataset_for_chromosome(donor_sites, acceptor_sites):

    min_len = min(len(donor_sites), len(acceptor_sites))
    
    #Cognates (Label 1)
    correct_pairs = []
    for i in range(min_len):
        donor_seq = donor_sites[i][1]
        acceptor_seq = acceptor_sites[i][1]
        
        #Valid sequences with same length 
        if donor_seq and acceptor_seq and len(donor_seq) == len(acceptor_seq):
            correct_pairs.append((donor_seq, acceptor_seq, 1))

    #Non-cognates (Label 0)
    num_pairs = len(correct_pairs) 
    if num_pairs == 0:
        return [], 0
        
    all_donors = [p[0] for p in correct_pairs]
    all_acceptors = [p[1] for p in correct_pairs]
    
    mismatched_pairs = []
    
    shift = random.randint(1, num_pairs - 1)
    for i in range(num_pairs):
        #Circular shift 
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
            continue
            
        #Creating dataset 
        chr_dataset, num_pairs = create_dataset_for_chromosome(donor_sites, acceptor_sites)
        
        print(f"  Label 1: {num_pairs} (chr{chr_num})")
        print(f"  Total of samples- chr{chr_num}: {len(chr_dataset)}")
        
        full_dataset.extend(chr_dataset)
        total_cognate_pairs += num_pairs
        
    random.shuffle(full_dataset)
    
    return full_dataset, total_cognate_pairs

#Changing to use on embedding layers 
def index_encode_single(sequence, seq_length):
    # A=0, C=1, G=2, T=3, N=4 (TO unknown bases)
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    # Array 1D of index
    encoded = np.zeros(seq_length, dtype=np.int32)
    default_index = mapping.get('N', 0)

    for i, base in enumerate(sequence):
        encoded[i] = mapping.get(base, default_index)
        
    return encoded

CHROMS_TO_PROCESS = list(range(1, 23))
print(f"chromosomes to process: {CHROMS_TO_PROCESS}")

aggregated_dataset, total_cognate_pairs = aggregate_all_chromosomes(CHROMS_TO_PROCESS)

if not aggregated_dataset:
    print("ERRO")
else:
    L = len(aggregated_dataset[0][0]) 
    total_samples = len(aggregated_dataset)

    #Arrays 2D (N, L)
    X_donor = np.zeros((total_samples, L), dtype=np.int32)
    X_acceptor = np.zeros((total_samples, L), dtype=np.int32)
    y = np.zeros((total_samples,), dtype=np.float32)
    
    print(f"\nSeq Dimension(L): {L}")
    print(f"Total of samples: {total_samples}")
    
    for i, (d_seq, a_seq, label) in enumerate(aggregated_dataset):
       
        #New function - coding by index
        X_donor[i] = index_encode_single(d_seq, L)
        X_acceptor[i] = index_encode_single(a_seq, L)
        y[i] = label

    np.save('X_donor_encoded.npy', X_donor)
    np.save('X_acceptor_encoded.npy', X_acceptor)
    np.save('y_labels.npy', y)

    print("\nsuccess")
    print(f"Shape de X_donor (Índices): {X_donor.shape}")
import numpy as np

class Symmetrical_Coupling_Helper:     
    ordered_indices : np.ndarray
    d_sys_ordered : list[int]
    l_list_ordered : list[int]
    fback_subchain_lengths : np.ndarray
    odd_end : bool
    sys_num : int
    interaction_num : int

    def __init__(self, d_sys_total):
        d_sys_total = np.array(d_sys_total)
        self.sys_num = len(d_sys_total)
        self.set_ordered_indices(self.sys_num)
        self.set_d_sys_ordered(d_sys_total)
        self.set_odd_end(self.sys_num)
        self.interaction_num = int(self.sys_num/2) + int(self.odd_end)
        # Function should be called explicitly to reduce constructor arguments
        #self.set_fback_subchain_lengths(l_list)

    def calc_l_list(taus, d_sys_total, delta_t):
        sys_num = len(d_sys_total)
        taus = np.array(taus)
        # N=2 case
        if sys_num == 2:
            return np.array(np.round(taus/delta_t, 0).astype(int))

        # In case of half taus specified, infer rest of taus by symmetry requirements
        if len(taus) == int(round((sys_num-1)/2)) and sys_num % 2 != 0:
            taus = np.append(taus, taus[::-1])
        elif len(taus) == int(round(sys_num/2)) and sys_num % 2 == 0:
            taus = np.append(taus, taus[len(taus)-2::-1])

        l_list=np.array(np.round(taus/delta_t, 0).astype(int)) #time steps between system and feedback
        # Check errors
        if not (l_list == l_list[::-1]).all():
            raise ValueError("Delay times tau list must be symmetric over reversal.")

        return l_list

    # indexed from right to left of the MPS
    def set_ordered_indices(self, sys_num):
        indices = np.arange(sys_num, dtype=int)
        half_point = int(sys_num/2)
        first_half = indices[:half_point]
        rev_end_half = indices[:half_point-1:-1]

        result = np.zeros(sys_num, dtype=int)
        result[:-1:2] = first_half
        result[1:-1:2] = rev_end_half[:-1]
        result[-1] = rev_end_half[-1]
        self.ordered_indices = result

    def set_d_sys_ordered(self, d_sys_total):
        self.d_sys_ordered = d_sys_total[self.ordered_indices]

    # Ordered from right to left in the MPS
    def set_fback_subchain_lengths(self, l_list):
        #self.l_list_ordered = l_list[self.ordered_indices]
        
        #special case of N=2
        if len(l_list) == 1:
            self.fback_subchain_num = 1
            self.fback_subchain_lengths = l_list
            return

        sys_num = len(l_list) + 1
        self.fback_subchain_num = int((sys_num+1)/2) # Take ceiling of half
        fback_subchain_lengths = np.zeros(self.fback_subchain_num, dtype=int)
        fback_subchain_lengths[1:] = (l_list[1::] + l_list[1::][::-1])[:self.fback_subchain_num-1] # Truncate the first, used at front and not doubled up. Addition compresses to half length

        # Check if need to halve the last case due to double counting middle l_list
        if sys_num % 2 != 0:
            fback_subchain_lengths[-1] /= 2 
        
        # Set the first value
        fback_subchain_lengths[0] = l_list[0]

        self.fback_subchain_lengths = fback_subchain_lengths

    def set_odd_end(self, sys_num):
        if sys_num % 2 == 0:
            self.odd_end = False
        else:
            self.odd_end = True

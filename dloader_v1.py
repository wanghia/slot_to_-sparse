import numpy as np
from os.path import isfile
from collections import defaultdict
from scipy.sparse import csr_matrix

import copy

import pdb


class GCDataLoader_v1:
    """
    slot_dict: save the sign dict of a slot
    signid2dis: a sign_id is discrete
    signid2slot: the slot of a sign id
    signid2sign: the sign of a sign id
    """
    def __init__(self, add_bias=False):
        # slot:sign:sign_id
        self.add_bias    = add_bias

        self.slot_dict   = defaultdict(dict)
        self.signid2sign = []
        self.signid2slot = []
        self.signid2dis  = []
        self.cnt         = 0
        self.first_load  = True

    def _slot_parser(self, j, da, cat):

        if cat == 'Categorical':
            is_descrete = True
            sign = da
            value = 1
        else:
            is_descrete = False
            sign = j
            if da.lower()=='nan':
                value = 0
                print("++++++++++++++++++++")
            else:
                value = float(da)

        slot = j

        if self.first_load:
            sign_id = self._get_and_create_sign_id(slot, sign, is_descrete)
        else:
            try:
                sign_id = self._get_sign_id(slot, sign)
            except:
                sign_id = -1
                print ("warning: drop inexist col")

        return slot, sign_id, value

    def _get_sign_id(self, slot, sign):
        return self.slot_dict[slot][sign]

    def _get_and_create_sign_id(self, slot, sign, is_descrete):
        d = self.slot_dict[slot]
        sign_id = d.get(sign)

        if is_descrete:
            fea_type = "Categorical"
        else:
            fea_type = "Numerical"
        if sign_id is None:
            sign_id = self.cnt
            d[sign] = sign_id
            self.signid2sign.append(sign)
            self.signid2slot.append(slot)
            self.signid2dis.append(fea_type)
            self.cnt += 1

        return sign_id

    def load(self, filename, dense=False, feat_type = list([])):

        with open(filename, 'r') as f:
            content = [x.strip() for x in f]


        vals = []
        col_ids = []
        row_ids = []

        for i, line in enumerate(content):
            row_vals = []
            row_col_ids = []

            d = [x.strip() for x in line.split(' ')]

            for j in range(len(d)):
                x = d[j]
                cat = feat_type[j]
                slot, sign_id, value = self._slot_parser(j,x,cat)
                if sign_id != -1:
                    row_vals.append(value)
                    row_col_ids.append(sign_id)

            vals.append(row_vals)
            col_ids.append(row_col_ids)
            row_ids.append([i] * len(row_col_ids))

        if self.add_bias:
            if self.first_load:
                self.bias_id = self.cnt
                self.cnt += 1

            for col, row, val in zip(col_ids, row_ids, vals):
                val.append(1)
                col.append(self.bias_id)
                row.append(row[-1])

        vals = [x for val in vals for x in val]
        col_ids = [x for col in col_ids for x in col]
        row_ids = [x for row in row_ids for x in row]

        if self.first_load:
            self.n_col = self.cnt

        n_row = len(content)

        if dense:
            tmp = self._dense_maker(n_row, self.n_col, row_ids, col_ids, vals)
        else:
            tmp = self._sparse_maker(n_row, self.n_col, row_ids, col_ids, vals)

        if self.first_load:
            self.first_load  = False
            self.data        = tmp
            self.slot_dict   = self.slot_dict

            self.signid2sign = np.array(self.signid2sign)
            self.signid2slot = np.array(self.signid2slot)
            self.signid2dis  = np.array(self.signid2dis)

            return self

        else:
            new_obj = copy.deepcopy(self)
            new_obj.data = tmp

            return new_obj

    def _dense_maker(self, nrow, ncol, row_ids, col_ids, vals):
        tmp = self._sparse_maker(nrow, ncol, row_ids, col_ids, vals).toarray()
        # tmp = np.zeros((n_row, n_col))

        # for i, (v, c_id) in enumerate(zip(vals, col_ids)):
            # tmp[i][c_id] = v

        return tmp

    def _sparse_maker(self, nrow, ncol, row_ids, col_ids, vals):

        tmp = csr_matrix((vals, (row_ids, col_ids)), shape=(nrow, ncol))
        return tmp






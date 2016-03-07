#include "fpstack.h"
#include <iostream>
#include <cstdlib>
#include <map>
#include <climits>
#include <vector>


void relative_item_ranks(double itemid1, double itemid2, 
        size_t *rank1, size_t *rank2, map<double, size_t> & item_map) {

    assert(itemid1 != itemid2);

    if (itemid1 == delimiter) {
        *rank1 = LONG_MAX;
        *rank2 = 0;
        return;
    }

    if (itemid2 == delimiter) {
        *rank2 = LONG_MAX;
        *rank1 = 0;
        return;
    }

    *rank1 = item_map[itemid1];
    *rank2 = item_map[itemid2];

    if(item_map[itemid1] == item_map[itemid2]) {
        if(itemid1 > itemid2){
            *rank1 = 0;
            *rank2 = 1;
        } else {
            *rank2 = 0;
            *rank1 = 1;

        }
    }
    else if (item_map[itemid1] < item_map[itemid2]) {
        *rank1 = 0;
        *rank2 = 1;
    }
    else {
        *rank2 = 0;
        *rank1 = 1;
    }

}

size_t appendSubTree(vector<double> & result,  size_t rpos, 
        vector<size_t>& rfreq, vector<double> & t, size_t tpos, vector<size_t> & tfreq) {
    size_t ndel = 0, nchar = 0;
    size_t pos = tpos;

    if(t[pos] == delimiter) {
        exit(1);
    }
    else {
        result[rpos] = t[pos];
        rfreq[rpos] = tfreq[pos];
        rpos++; pos++;
        nchar++;
    }

    while(ndel < nchar){
        if(t[pos] == delimiter)
            ndel++;
        else
            nchar++;

        result[rpos] = t[pos];
        rfreq[rpos] = tfreq[pos];
        rpos++; pos++;
    }

    return (pos-tpos);

}


size_t fpmerge(vector<double> &t1, size_t size1, vector<size_t> &freq1, vector<double> & t2, size_t size2, 
    vector<size_t> &freq2, vector<double> &result, vector<size_t> & resultFreq, map<double, size_t> &item_map)
{

    size_t index1 = 0, index2 = 0, index = 0;
    size_t subtreeSize = 0;

    size_t rank1, rank2, i;

    if(size1 == 0) {
        cout << "Transaction 1 empty "<<endl;
        for (i = 0; i < size2; i++) {
            resultFreq[i] = freq2[i];
            result[i] = t2[i];
        }
        return size2;
    }

    if(size2 == 0) {
        cout << "Transaction 2 empty "<<endl;
        for (i = 0; i < size1; i++) {
            resultFreq[i] = freq1[i];
            result[i] = t1[i];
        }
        return size1;
    }
    while(index1 < size1 && index2 < size2) {

        if(t1[index1] == t2[index2]) {
            /* same labels */
            result[index] = t1[index1];
            if(t1[index1] == DEL)
                /* Its a delimiter */
                resultFreq[index] = 1;
            else
                resultFreq[index] = freq1[index1] + freq2[index2];

            index1++; index2++; index++;

        } else {
            /* no matching labels */
            relative_item_ranks(t1[index1], t2[index2], &rank1, &rank2, item_map);

            if(rank1 < rank2) {
                subtreeSize = appendSubTree(result, index, resultFreq, t1, index1, freq1);
                index1 = index1 + subtreeSize;


            } else {
                subtreeSize = appendSubTree(result, index, resultFreq, t2, index2, freq2);
                index2 = index2 + subtreeSize;
            }

            index = index+ subtreeSize;
        }
    } /* endof while */

    while (index1 < size1) {
        subtreeSize = appendSubTree(result, index, resultFreq, t1, index1, freq1);
        index1 = index1 + subtreeSize;
        index = index+ subtreeSize;
    }

    while (index2 < size2) {
        subtreeSize = appendSubTree(result, index, resultFreq, t2, index2, freq2);
        index2 = index2 + subtreeSize;
        index = index+ subtreeSize;
    }
    return index;
}

#if 0
int if_exists(trans label, trans* label_list, int size ) {
    // KA : do this with hash map ??
    for(int i=0; i <size; i++){
        if(label_list[i] == label) 
            return 1;
    }
    return 0;
}
#endif

int64_t prune_local_prefix_tree(trans* orig_tree, int64_t tree_size, int64_t *freq1,
        trans* pruned_tree, int64_t* pruned_freq, trans* label_list, int64_t
        list_size){
#if 0
    if(!orig_tree or !freq1 or tree_size == 0) { 
        cout << "Cannot prune a  empty tree \n";
        return -1;
    }
    if(!pruned_tree or !pruned_freq){
        cout << "Please allocate memory for pruned tree\n";
        return -1;
    }

    if(!label_list){
        cout << "No lables assigned to this process, returning empty tree \n";
        pruned_tree = NULL;
        pruned_freq = NULL;
        return 0;
    }

    trans label, tmp;
    int64_t label_freq;
    stack s;

    /* copy root to result and on top of stack */
    pruned_tree[0] = ROOT;
    pruned_freq[0] = freq1[0];
    int64_t pruned_size = 1;
    s.push(ROOT, freq1[0]);

    /* Start after Root of tree*/
    for(int64_t i=1; i < tree_size; i++) {

        label = orig_tree[i];
        label_freq = freq1[i];

        if(label != delimiter){
            s.push(label, label_freq);
        }
        else{ 
            /* next label in tree is a delimiter
             * Find what is on top of stack
             */
            tmp = s.peek_label();

            if(tmp == ROOT)
            {
                // add delimiter to result if stack top is ROOT
                pruned_tree[pruned_size] = label;
                pruned_freq[pruned_size] = freq1[i];
                pruned_size++;
            }
            else if(if_exists(tmp, label_list, list_size)) {
                /* empty the stacks and add everythign 
                 * including current delimiter to result */
                s.push(label, label_freq);
                pruned_size = s.pop_to_array(pruned_tree, pruned_freq, pruned_size);

            }else{
                //pop the label on top and discard
                s.pop();
            }

        }// end of "if delimiter"

    }// end of for loop/original tree

    return pruned_size;
#endif
}


#include "fpheader.h"
#include <mpi.h>

struct node {
  trans label;
  int64_t freq;
  struct node* next;
  int64_t padding;
};


class mystack {

  // Top poinst to fisrt node in linked list
  // Top = NULL indociates empty mystack
  node *top;
  int64_t num_items;

  public:
  
  mystack() {top = NULL; num_items = 0;}
  ~mystack(){}

  /**** PUSH *****/
  void push(trans newlabel, int64_t label_freq){

    node* newnode = new node;
//    node* newnode;
//    MPI_Alloc_mem(sizeof(struct node), MPI_INFO_NULL, &newnode);
    assert(newnode);
    if(!newnode){
      std::cout << " could not allocate memnory for new node";
    }
    newnode->label = newlabel;
    newnode->freq = label_freq;
    newnode->next = NULL;
    if( top != NULL) {
      //Stack is not empty
      newnode->next = top;
    }
    top = newnode;
    num_items++;
  }

  /**** POP *****/
  trans pop() {
    if(top  == NULL) {
      // mystack is empty
      cout << "Nothing to pop";
      return -1;
    }

    node *temp = top;
    top = top->next;

    trans label = temp->label;
    temp->next = NULL;
    delete(temp);
    num_items--;
    
    return label;
  }

  /**** take a PEEK the mystack top *****/
  trans peek_label() {
    if(top  == NULL) {
      // mystack is empty
      cout << "Nothing to pop";
      return -1;
    }

    return (top->label);
  }

  /* Pops the mystack and puts elments into an array */
  /* returns the final size of array */
  int64_t pop_to_array(trans* array, int64_t* freq, int64_t cur_size) {

    int64_t num_items_to_add = this->num_items - 1; // do not pop the root
    int64_t total_items = cur_size + num_items_to_add;

    for(int64_t i=total_items-1; i >= cur_size; i--) {

      node* curnode = this->top;
      array[i] =curnode->label;
      freq[i] = curnode->freq;
      this->pop();
    }
    return total_items;
  }

  /**** Size of mystack *****/
  int64_t size(){
    return num_items;
  }

  /**** is staxk empty ****/
  int isEmpty() {
    if(top == NULL){
      assert(num_items == 0);
      return 1;
    }
    else
      return 0;
  }
};

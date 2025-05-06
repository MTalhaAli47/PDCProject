#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <time.h>
#include <cmath>

using namespace std;

//Helper functions
vector<int> decode_permutation(int id, int n, vector<int>& f){
    //Build the remainder table
    vector<int> remainder(n,1);
    for (int i=1;i<n;i++){
        remainder[i] = remainder[i-1]+1;
    }

    //Build permutation vector
    vector<int> permutation(n);

    //Compute the permutation
    for (int i=0;i<n;i++){
        int block_size = f[(n-1) - i];
        int j = id/block_size;

        permutation[i] = remainder[j];
        remainder.erase(remainder.begin() + j);
        id = id%block_size;
    }

    //Return permutation
    return permutation;
};

int encode_permutation(vector<int>& p, int n, vector<int>& f){
    //Build the remainder table
    vector<int> remainder(n,1);
    for (int i=1;i<n;i++){
        remainder[i] = remainder[i-1]+1;
    }

    //Initialize id
    int id=0;

    //Compute the permutation
    for (int i=0;i<n;i++){
        int x = p[i];
        int j = distance(remainder.begin(),find(remainder.begin(),remainder.end(),x));
        id += j * f[(n-1) - i];

        remainder.erase(remainder.begin() + j);
    }

    //Return id
    return id;
};

vector<int> swap(const vector<int>& v, int i){
    vector<int>p = v;
    int j = find(v.begin(), v.end(), i) - v.begin();
    if(j==v.size()-1){
        return v;
    }

    std::swap(p[j],p[j+1]);
    return p;
}

int r(const vector<int>& v){
    for(int i=v.size();i>0;i--){
        if(v[i-1]!=i){
            return i;
        }
    }
    return 0;
}

vector<int> find_position(const vector<int>& v, int t,const vector<int>& id){
    if(t==2 && swap(v,t)==id){
        return swap(v,t-1);
    }
    else if(v[v.size()-2]==t || v[v.size()-2]==v.size()-1){
        int j=r(v);
        return swap(v,j);
    }
    else{
        return swap(v,t);
    }
};

//Main function
int main(){
    //Setting the number of threads
    int num_threads=8;
    omp_set_num_threads(num_threads);

    //First input network size
    int n=1;
    while(n<2){
        cout<<"Enter the size of the network (greater than 2): ";
        cin>>n;
    }
    cout<<endl;
    int trees=n-1;

    //Next, calculate the total number of vertices/nodes, and compute an array of factorials
    int vertices=1;
    vector<int> factorial_arr(n+1,1);
    for (int i=2;i<=n;i++){
        vertices *= i;
        factorial_arr[i] = factorial_arr[i-1]*i;
    }

    //Finally, intialize a 2d array to store the parent ids
    vector<vector<int>> parent_ids(vertices, vector<int>(trees,0));
    vector<int> id(n);
    iota(id.begin(),id.end(),1);

    //Start time
    double start, end;
    start=omp_get_wtime();

    //Process each vertix (except for root vertices)
    for (int i=2; i<=vertices; i++){
        //Decode the vertix permutation
        vector<int> decoded = decode_permutation(i-1,n,factorial_arr);

        //For each tree...
        for(int j=1; j<=trees; j++){
                int vn=decoded[n-1];
                vector<int> parent;

                //Check conditions on what swap applies to this
                if(vn==n){
                    if(j!=n-1){
                        parent = find_position(decoded,j,id);
                    }
                    else{
                        //Case 2
                        parent = swap(decoded,decoded[n-2]);
                    }
                }
                else if(vn==n-1 && decoded[n-2]==n && swap(decoded,n)!=id){
                    if(j==1){
                        //Case 3
                        parent = swap(decoded,n);
                    }
                    else{
                        //Case 4
                        parent = swap(decoded,j-1);
                    }
                }
                else{
                    if(vn==j){
                        //Case 5
                        parent=swap(decoded,n);
                    }
                    else{
                        //Case 6
                        parent = swap(decoded,j);
                    }
                }

                //Encode the vertix permutation back into an id, and store it in the list
                parent_ids[i-1][j-1] = encode_permutation(parent,n,factorial_arr)+1;
            }
    }

    //End time
    end=omp_get_wtime();

    // Print final output
    /*
    for (int i = 0; i < vertices; i++) {
        cout<<i+1<<" -> [ ";
        for (int j = 0; j < trees; j++) {
            cout << parent_ids[i][j];
            if(j!=trees-1){
                cout<<", ";
            }
            else{
                cout<<" ";
            }
        }
        cout << "]" << endl;
    }
    */

    double time_taken=end-start;
    cout<<"Time taken for scalar execution => "<<time_taken<<"s"<<endl;
    
    cout<<endl;
    return 0;
}
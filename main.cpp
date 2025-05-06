#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <mpi.h>

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
int main(int argc, char** argv){
    //Setting the number of threads
    int num_threads=8;
    omp_set_num_threads(num_threads);
    
    //Initializing MPI
    MPI_Init(&argc, &argv);

    //Start time
    double start, end;
    start=MPI_Wtime();

    //Setting rank and size
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Each process gets a unique rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Total number of processes

    //Initializing variables
    int n=1;

    //Master MPI
    if(world_rank==0){
        //First input network size
        while(n<2){
            cout<<"Enter the size of the network (greater than 2): ";
            cin>>n;
        }
        cout<<endl;
    }

    //Broadcast the shared variables across the system
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int trees=n-1;
    int vertices=1;
    vector<int> factorial_arr(n + 1, 1);

    //Next, calculate the total number of vertices/nodes, and compute an array of factorials
    for (int i=2;i<=n;i++){
        vertices *= i;
        factorial_arr[i] = factorial_arr[i-1]*i;
    }

    //MPI Work division
    int total = trees * (vertices-1);

    int chunk_size = total/world_size;
    int remainder = total % world_size;
    int start_k = world_rank * chunk_size + min(world_rank,remainder);
    int end_k = start_k + chunk_size + (world_rank < remainder ? 1 : 0);

    //Finally, intializelocals
    vector<int> id(n);
    iota(id.begin(),id.end(),1);
    int my_chunk_size = end_k - start_k; // How many k values this process will compute
    vector<int> local_ids(my_chunk_size, 0); // Flat storage for encoded results

    //Process each vertix (except for root vertices)
    #pragma omp parallel for
    for(int k=start_k; k<end_k; k++){
        //Calculating indices
        int i = k/trees + 2;
        int j = k%trees + 1;

        //Initialize permutations
        vector<int> decoded = decode_permutation(i-1,n,factorial_arr);
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
        //No dependancy here
        local_ids[k - start_k] = encode_permutation(parent,n,factorial_arr)+1;
    }

    //Retrieving results
    vector<int> global_ids(trees * (vertices - 1));
    vector<int> count(world_size);
    vector<int> location(world_size);

    if(world_rank==0){
        for (int i = 0; i < world_size; ++i) {
            count[i] = chunk_size + (i < remainder ? 1 : 0);
            if (i == 0)
                location[i] = 0;
            else
                location[i] = location[i - 1] + count[i - 1];
        }
    }

    MPI_Gatherv(local_ids.data(),local_ids.size(),MPI_INT,global_ids.data(),count.data(),location.data(),MPI_INT,0,MPI_COMM_WORLD);

    if (world_rank == 0) {
        vector<vector<int>> parent_ids(vertices, vector<int>(trees, 0));
        for (int k = 0; k < total; ++k) {
            int i = k / trees + 2;
            int j = k % trees + 1;
            parent_ids[i - 1][j - 1] = global_ids[k];
        }
        
        // Print final output
        /*for (int i = 0; i < vertices; i++) {
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
        }*/

        //End time
        end=MPI_Wtime();

        double time_taken=end-start;
        cout<<"Time taken for parallel execution => "<<time_taken<<"s"<<endl;
    }

    //Finalizations
    MPI_Finalize();
    
    cout<<endl;
    return 0;
}
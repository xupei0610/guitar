# Mocap Data

This dataset can be used under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License.

![](camera_deploy.png)

All data were collected from a professional guitarist. They are general guitar-playing motions rather than playing any specific songs. We deployed 18 infrared cameras for motion capture, 7 of which have a resolution of 9MP and 11 have a resolution of 4MP. All cameras are the products from [NOKOV](https://www.nokov.com/). 

We thank [NOKOV](https://www.nokov.com/) for their supporting of motion capture solutions, and Hao Wang, the subject guitarist, for his performance. 

If you use this dataset, please consider citing the paper:

    @article{adaptnet,
        author = {Xu, Pei and Wang, Ruocheng},
        title = {Synchronize Dual Hands for Physics-Based Dexterous Guitar Playing},
        booktitle = {SIGGRAPH Asia 2024 Conference Papers (SA Conference Papers '24)},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        year = {2024},
        doi = {10.1145/3680528.3687692}
    }


## Papers using This Dataset

- [Synchronize Dual Hands for Physics-Based Dexterous Guitar Playing](https://pei-xu.github.io/guitar)


If you use this dataset, please contact me to add your work in this list.


## Data Format

All data items are provided in CSV format with markers defined as the following picture.

![](marker.png)

For each marker, we provide its 3D position in XYZ order.
Note that due to occlusion, some marker positions are not available to reconstruct. For those markers, we use `nan` in the CSV files.

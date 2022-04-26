# Gesture-based System for Next Generation Natural and Intuitive Interfaces

We developed a novel and trainable gesture-based system for next generation intelligent interfaces. The system requires a non-contact depth sensing device such as an RGB-D camera for user input. The camera records the user's static hand pose and palm center dynamic motion trajectory. Both static pose and dynamic trajectory are used independently to provide commands to the interface. With regard to the palm center trajectory, we collected 3D sketch data from multiple users pertaining to various classes of sketches/symbols. For more details, please refer to our <a href="https://doi.org/10.1017/S0890060418000045">publication</a> in Artificial Intelligence for Engineering Design, Analysis and Manufacturing journal.

This repository includes a zipped folder consisting of the complete 3D sketch dataset collected from ten users consisting of 40 different symbols. An image has been included for all the 40 symbols. The data was collected using depth-sensing camera SoftKinetic DepthSense DS325.

Each subfolder contains sketch samples from one of the following domains:
1. Arabic numerals
2. English alphabets
3. Simulation symbols
4. CAD primitives

Every domain has 10 different classes of sketches, recorded by 10 users.
The first three lines in the data file specifies the domain id (1-4), class id (1-10), and user id (1-10).
Following the above information, the column title are provided, which are 'x', 'y', 'z', and 't'.
Each row corresponds to a data point on the sketch (in order), specifying its 3D spatial location (x, y, z), and time stamp in milliseconds (t).

If you use our dataset for your work, please cite:
> Huang, J., Jaiswal, P., & Rai, R. (2018). Gesture-based system for next generation natural and intuitive interfaces. Artificial Intelligence for Engineering Design, Analysis and Manufacturing, 1-15. doi:10.1017/S0890060418000045


*****************************************************************
### Owner:
	MAD Lab
	Department of Mechanical and Aerospace Engineering
	University at Buffalo, Buffalo, NY - 14260
	http://madlab.eng.buffalo.edu/
*****************************************************************

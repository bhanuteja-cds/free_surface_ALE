
fac = 10;
lc1 = 0.05;
xmax = 0.1*fac;
ymax = 0.015*fac;
zmax = 0.04*fac;
// wavh = 0.00425*fac;
wavh = 0.004*fac;

// Point(1) = {0,0,0,lc1*5};
// Point(2) = {xmax,0,0,lc1*5};
Point(1) = {0,0,0,lc1};
Point(2) = {xmax,0,0,lc1};
For t In {xmax:-0.001*fac:-0.005*fac}
Point(newp) = {t, ymax+wavh*Sin(2*Pi*t/xmax), 0, lc1};
EndFor
pend1=newp-1;
Line(1) = {1,2};
Line(2) = {2,3};
Spline(3) = {3:pend1};
Line(4) = {pend1,1};
Line Loop(1) = {1:4};
Plane Surface(1) = {1};
//fluback
//===================================================================
p1=newp;
// Point(p1) = {0,0,zmax,lc1*5};
// Point(newp) = {xmax,0,zmax,lc1*5};
Point(p1) = {0,0,zmax,lc1};
Point(newp) = {xmax,0,zmax,lc1};
For t In {xmax:-0.001*fac:-0.005*fac}
Point(newp) = {t,ymax+wavh*Cos(2*Pi*t/xmax), zmax, lc1};
EndFor
pend2 = newp-1;
l1=newl;
Line(l1) = {p1,p1+1};
Line(newl) = {p1+1,p1+2};
Spline(newl) = {p1+2:pend2};
l2=newl;
Line(l2) = {pend2,p1};
Line Loop(2) = {l1:l2};
Plane Surface(2) = {2};
//flufront
//==================================================================
Line(newl) = {1,pend1+1};
Line(newl) = {2,pend1+2};
Line(newl) = {3,pend1+3};
Line(newl) = {pend1,pend2};
//===============================
Line Loop(3) = {9,-8,-12,4};
Plane Surface(3) = {3};  //fluleft
Line Loop(4) = {9,5,-10,-1};
Plane Surface(4) = {4}; //flubottom
Line Loop(5) = {10,6,-11,-2};
Plane Surface(5) = {5};  //fluright
Line Loop(6) = {-7,-11,3,12};
Ruled Surface(6) = {6};  //flutop
//=================================
Surface Loop(1) = {1,2,3,4,5,6};
Volume(1) = {1};

// Point(newp) = {0.0092, 0.0022, 0, lc1};
// Line(newl) = {3, newp-1};
// Line{newl-1} In Surface{1};
// 
// Point(newp) = {0.01, 0.0025, 0.0007, lc1};
// Line(newl) = {3, newp-1};
// Line{newl-1} In Surface{5};
// 
// Point(newp) = {0.01, 0.0023, 0.003, lc1};
// Line(newl) = {26, newp-1};
// Line{newl-1} In Surface{5};
// 
// Point(newp) = {0.0093, 0.0025, 0.004, lc1};
// Line(newl) = {26, newp-1};
// Line{newl-1} In Surface{2};
// 
// Point(newp) = {0, 0.0023, 0.003, lc1};
// Line(newl) = {46, newp-1};
// Line{newl-1} In Surface{3};
// 
// Point(newp) = {0.001, 0.0023, 0.004, lc1};
// Line(newl) = {46, newp-1};
// Line{newl-1} In Surface{2};
// 
// Point(newp) = {0.001, 0.0023, 0.0, lc1};
// Line(newl) = {23, newp-1};
// Line{newl-1} In Surface{1};
// 
// Point(newp) = {0, 0.0025, 0.0008, lc1};
// Line(newl) = {23, newp-1};
// Line{newl-1} In Surface{3};
//================================
// Transfinite Line{1:4, 5:8, 9:12} = 12;
// Transfinite Surface {1} = {1,2,3,23};
// Transfinite Surface {2} = {24, 25, 26, 46};
// Transfinite Surface {3} = {1, 23, 46, 24};
// Transfinite Surface {4} = {24,25,2,1};
// Transfinite Surface {5} = {25,2,3,26};
// Transfinite Surface {6} = {46,26,3,23};
// Transfinite Volume{1} = {1,2,3,23,24,25,26,46};

Physical Surface(101)={1};
Physical Surface(102)={2};
Physical Surface(103)={3};
Physical Surface(104)={4};
Physical Surface(105)={5};
Physical Surface(150)={6};
Physical Volume(250) = {1}; 

// Mesh.RecombinationAlgorithm = 2;
// Mesh.RecombineAll = 1;
// Mesh.Hexahedra = 1;

// Recombine Surface {1:6};
// Recombine Volume {1};
//=================================

// Field[1] = Attractor;
// // Field[1].NodesList = {1, 2, 53, 54, 3, 52, 55, 104};
// Field[1].NodesList = {1, 2, 53, 54};
// // Field[1].EdgesList = {1};
// Field[2] = Threshold;
// Field[2].IField = 1;
// Field[2].LcMin = lc1/10;
// Field[2].LcMax = lc1;
// Field[2].DistMin = 0.0015;
// Field[2].DistMax = 0.01;
// 
// Field[7] = Min;
// Field[7].FieldsList = {2};
// Background Field = 7;


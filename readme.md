## compile

run
```bash
./Allwmake -j
```


## usage
in **constant/chemistryProperties**

```bash
chemistryType
{
    solver          ode;
    method          pure; //key line
}
```

optimizedODE  true;

in **system/controlDict**

```bash
libs ("libpureChemistryModel.so");
```


### check the Sandia tutorial case for details!
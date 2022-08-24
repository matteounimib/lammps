// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_duplicate.h"

#include "atom.h"
#include "atom_vec.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "lattice.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "random_park.h"
#include "region.h"
#include "update.h"
#include "utils.h"
#include "group.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <vector>
#include <algorithm>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;
using namespace std;

enum{ATOM,MOLECULE};
enum{DIST_UNIFORM,DIST_GAUSSIAN};

#define EPSILON 1.0e6

/* ---------------------------------------------------------------------- */

FixDuplicate::FixDuplicate(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), idregion(nullptr), idrigid(nullptr),
  idshake(nullptr), onemols(nullptr), molfrac(nullptr), coords(nullptr), imageflags(nullptr),
  fixrigid(nullptr), fixshake(nullptr), random(nullptr)
{
  if (narg < 8) error->all(FLERR,"Illegal fix deposit command");

  restart_global = 1;
  time_depend = 1;

  // required args

  ninsert = utils::inumeric(FLERR,arg[3],false,lmp);
  ntype = utils::inumeric(FLERR,arg[4],false,lmp);
  group_child = utils::inumeric(FLERR, arg[5], false, lmp);
  nfreq = utils::inumeric(FLERR,arg[6],false,lmp);
  seed = utils::inumeric(FLERR,arg[7],false,lmp);
  prob = utils::numeric(FLERR,arg[8],false,lmp);
  radius = utils::numeric(FLERR,arg[9],false,lmp);
  // in case it will uses groups names
  // group_source = group->find(arg[4]);
  // group_source = utils::inumeric(FLERR, arg[4], false, lmp);
  groupbit_source = group->bitmask[ntype];
  groupbit_child = group->bitmask[group_child];
  // cout << "group parent: " << ntype << " " << groupbit_source << endl;
  // cout << "group child : " << group_child << " " << groupbit_child << endl;
  if (prob < 0.0 || prob > 1.0) error->all(FLERR, "Probability must be in [0,1] interval");
  if (radius <= 0.0) error->all(FLERR, "Radius must be greater than 0.0");
  if (seed <= 0) error->all(FLERR,"Illegal fix deposit command");

  // read options from end of input line

  options(narg-10,&arg[10]);

  // error check on type

  if (mode == ATOM && (ntype <= 0 || ntype > atom->ntypes))
    error->all(FLERR,"Invalid atom type (parent) in fix duplicate command");

  if (mode == ATOM && (group_child <= 0 || group_child > atom->ntypes))
    error->all(FLERR,"Invalid atom type (child) in fix duplicate command");

  // error checks on region and its extent being inside simulation box

  // if (!iregion) error->all(FLERR,"Must specify a region in fix deposit");
  // if (iregion->bboxflag == 0)
  //   error->all(FLERR,"Fix deposit region does not support a bounding box");
  // if (iregion->dynamic_check())
  //   error->all(FLERR,"Fix deposit region cannot be dynamic");

  // xlo = iregion->extent_xlo;
  // xhi = iregion->extent_xhi;
  // ylo = iregion->extent_ylo;
  // yhi = iregion->extent_yhi;
  // zlo = iregion->extent_zlo;
  // zhi = iregion->extent_zhi;

  if (domain->triclinic == 0) {
    if (xlo < domain->boxlo[0] || xhi > domain->boxhi[0] ||
        ylo < domain->boxlo[1] || yhi > domain->boxhi[1] ||
        zlo < domain->boxlo[2] || zhi > domain->boxhi[2])
      error->all(FLERR,"Deposition region extends outside simulation box");
  } else {
    if (xlo < domain->boxlo_bound[0] || xhi > domain->boxhi_bound[0] ||
        ylo < domain->boxlo_bound[1] || yhi > domain->boxhi_bound[1] ||
        zlo < domain->boxlo_bound[2] || zhi > domain->boxhi_bound[2])
      error->all(FLERR,"Deposition region extends outside simulation box");
  }

  // error check and further setup for mode = MOLECULE

  if (atom->tag_enable == 0)
    error->all(FLERR,"Cannot use fix_deposit unless atoms have IDs");

  if (mode == MOLECULE) {
    for (int i = 0; i < nmol; i++) {
      if (onemols[i]->xflag == 0)
        error->all(FLERR,"Fix deposit molecule must have coordinates");
      if (onemols[i]->typeflag == 0)
        error->all(FLERR,"Fix deposit molecule must have atom types");
      if (ntype+onemols[i]->ntypes <= 0 ||
          ntype+onemols[i]->ntypes > atom->ntypes)
        error->all(FLERR,"Invalid atom type in fix deposit mol command");

      if (atom->molecular == Atom::TEMPLATE && onemols != atom->avec->onemols)
        error->all(FLERR,"Fix deposit molecule template ID must be same "
                   "as atom_style template ID");
      onemols[i]->check_attributes(0);

      // fix deposit uses geoemetric center of molecule for insertion

      onemols[i]->compute_center();
    }
  }

  if (rigidflag && mode == ATOM)
    error->all(FLERR,"Cannot use fix deposit rigid and not molecule");
  if (shakeflag && mode == ATOM)
    error->all(FLERR,"Cannot use fix deposit shake and not molecule");
  if (rigidflag && shakeflag)
    error->all(FLERR,"Cannot use fix deposit rigid and shake");

  // setup of coords and imageflags array

  if (mode == ATOM) natom_max = 1;
  else {
    natom_max = 0;
    for (int i = 0; i < nmol; i++)
      natom_max = MAX(natom_max,onemols[i]->natoms);
  }
  memory->create(coords,natom_max,3,"deposit:coords");
  memory->create(imageflags,natom_max,"deposit:imageflags");

  // setup scaling

  double xscale,yscale,zscale;
  if (scaleflag) {
    xscale = domain->lattice->xlattice;
    yscale = domain->lattice->ylattice;
    zscale = domain->lattice->zlattice;
  }
  else xscale = yscale = zscale = 1.0;

  // apply scaling to all input parameters with dist/vel units

  if (domain->dimension == 2) {
    lo *= yscale;
    hi *= yscale;
    rate *= yscale;
  } else {
    lo *= zscale;
    hi *= zscale;
    rate *= zscale;
  }
  deltasq *= xscale*xscale;
  nearsq *= xscale*xscale;
  vxlo *= xscale;
  vxhi *= xscale;
  vylo *= yscale;
  vyhi *= yscale;
  vzlo *= zscale;
  vzhi *= zscale;
  xmid *= xscale;
  ymid *= yscale;
  zmid *= zscale;
  sigma *= xscale; // same as in region sphere
  tx *= xscale;
  ty *= yscale;
  tz *= zscale;

  // find current max atom and molecule IDs if necessary
  if (idnext) find_maxid();

  // random number generator, same for all procs
  // warm up the generator 30x to avoid correlations in first-particle
  // positions if runs are repeated with consecutive seeds

  random = new RanPark(lmp,seed);
  for (int ii=0; ii < 30; ii++) random->uniform();

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;
  nfirst = next_reneighbor-nfreq;
  ninserted = 0;
}

/* ---------------------------------------------------------------------- */

FixDuplicate::~FixDuplicate()
{
  delete random;
  delete [] molfrac;
  delete [] idrigid;
  delete [] idshake;
  delete [] idregion;
  memory->destroy(coords);
  memory->destroy(imageflags);
}

/* ---------------------------------------------------------------------- */

int FixDuplicate::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */
void FixDuplicate::init()
{
  // set index and check validity of region

  // iregion = domain->get_region_by_id(idregion);
  // if (!iregion) error->all(FLERR,"Region ID {} for fix deposit does not exist", idregion);

  // if rigidflag defined, check for rigid/small fix
  // its molecule template must be same as this one

  fixrigid = nullptr;
  if (rigidflag) {
    fixrigid = modify->get_fix_by_id(idrigid);
    if (!fixrigid) error->all(FLERR,"Fix deposit rigid fix ID {} does not exist", idrigid);
    int tmp;
    if (onemols != (Molecule **) fixrigid->extract("onemol",tmp))
      error->all(FLERR, "Fix deposit and rigid fix are not using the same molecule template ID");
  }

  // if shakeflag defined, check for SHAKE fix
  // its molecule template must be same as this one

  fixshake = nullptr;
  if (shakeflag) {
    fixshake = modify->get_fix_by_id(idshake);
    if (!fixshake) error->all(FLERR,"Fix deposit shake fix ID {} does not exist", idshake);
    int tmp;
    if (onemols != (Molecule **) fixshake->extract("onemol",tmp))
      error->all(FLERR,"Fix deposit and fix shake are not using the same molecule template ID");
  }

  // for finite size spherical particles:
  // warn if near < 2 * maxrad of existing and inserted particles
  //   since may lead to overlaps
  // if inserted molecule does not define diameters,
  //   use AtomVecSphere::create_atom() default radius = 0.5

  if (atom->radius_flag) {
    double *radius = atom->radius;
    int nlocal = atom->nlocal;

    double maxrad = 0.0;
    for (int i = 0; i < nlocal; i++)
      maxrad = MAX(maxrad,radius[i]);

    double maxradall;
    MPI_Allreduce(&maxrad,&maxradall,1,MPI_DOUBLE,MPI_MAX,world);

    double maxradinsert = 0.0;
    if (mode == MOLECULE) {
      for (int i = 0; i < nmol; i++) {
        if (onemols[i]->radiusflag)
          maxradinsert = MAX(maxradinsert,onemols[i]->maxradius);
        else maxradinsert = MAX(maxradinsert,0.5);
      }
    } else maxradinsert = 0.5;

    double separation = MAX(2.0*maxradinsert,maxradall+maxradinsert);
    if (sqrt(nearsq) < separation && comm->me == 0)
      error->warning(FLERR,"Fix deposit near setting < possible overlap separation {}",separation);
  }
}

/* ---------------------------------------------------------------------- */

void FixDuplicate::setup_pre_exchange()
{
  if (ninserted < ninsert) next_reneighbor = nfirst + ((update->ntimestep - nfirst)/nfreq)*nfreq + nfreq;
  else next_reneighbor = 0;
}

/* ----------------------------------------------------------------------
   perform particle insertion
------------------------------------------------------------------------- */
void FixDuplicate::generate_positions(struct added_coord *out){
  int i,m,nlocalprev,imol,natom,flag,flag_prob;
  double coord[3],delx,dely,delz,rsq;
  double r[3],vnew[3],rotmat[3][3],quat[4];
  flag_prob = 0;
  int added = 0;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double **x = atom->x;
  double** localCoord = {};
  vector<double*> localVel = {};
  double *sublo,*subhi;
  int dimension = domain->dimension;
  out->coord = (double**) malloc(sizeof(double*));
  out->vel = (double**) malloc(sizeof(double*));
  for (int q = 0; q < nlocal; q++){
    if(mask[q] & groupbit_source){
      flag_prob = 0;
      if (random->uniform() < prob) flag_prob = 1;
      xlo = x[q][0] - radius;
      ylo = x[q][1] - radius;
      zlo = x[q][2] - radius;
      xhi = x[q][0] + radius;
      yhi = x[q][1] + radius;
      zhi = x[q][2] + radius;
      // just return if should not be called on this timestep
      if (next_reneighbor != update->ntimestep) {
        out->added = added;
        out->coord = localCoord;
      }

      // clear ghost count (and atom map) and any ghost bonus data
      //   internal to AtomVec
      // same logic as beginning of Comm::exchange()
      // do it now b/c inserting atoms will overwrite ghost atoms

      if (atom->map_style != Atom::MAP_NONE) atom->map_clear();
      atom->nghost = 0;
      atom->avec->clear_bonus();

      // compute current offset = bottom of insertion volume

      double offset = 0.0;
      if (rateflag) offset = (update->ntimestep - nfirst) * update->dt * rate;
      // moved
      if (domain->triclinic == 0) {
        sublo = domain->sublo;
        subhi = domain->subhi;
      } else {
        sublo = domain->sublo_lamda;
        subhi = domain->subhi_lamda;
      }

      // find current max atom and molecule IDs if necessary

      // if (!idnext) find_maxid();

      // attempt an insertion until successful
      // moved
      // int dimension = domain->dimension;

      int attempt = 0;
      while (attempt < maxattempt) {
        attempt++;

        // choose random position for new particle within region
        if (distflag == DIST_UNIFORM) {
          // do {
          coord[0] = xlo + random->uniform() * (xhi-xlo);
          coord[1] = ylo + random->uniform() * (yhi-ylo);
          coord[2] = zlo + random->uniform() * (zhi-zlo);
          // } while (iregion->match(coord[0],coord[1],coord[2]) == 0);
        } else if (distflag == DIST_GAUSSIAN) {
          // do {
          coord[0] = xmid + random->gaussian() * sigma;
          coord[1] = ymid + random->gaussian() * sigma;
          coord[2] = zmid + random->gaussian() * sigma;
          // } while (iregion->match(coord[0],coord[1],coord[2]) == 0);
        } else error->all(FLERR,"Unknown particle distribution in fix duplicate");

        // adjust vertical coord by offset

        if (dimension == 2) coord[1] += offset;
        else coord[2] += offset;

        // remap coord for PBC
        domain->remap(coord);
        // coords = coords of all atoms
        // for molecule, perform random rotation around center pt
        // apply PBC so final coords are inside box
        // also modify image flags due to PBC

        if (mode == ATOM) {
          natom = 1;
          coords[0][0] = coord[0];
          coords[0][1] = coord[1];
          coords[0][2] = coord[2];
          imageflags[0] = ((imageint) IMGMAX << IMG2BITS) |
            ((imageint) IMGMAX << IMGBITS) | IMGMAX;
        } else {
          double rng = random->uniform();
          imol = 0;
          while (rng > molfrac[imol]) imol++;
          natom = onemols[imol]->natoms;
          if (dimension == 3) {
            if (orientflag) {
              r[0] = rx;
              r[1] = ry;
              r[2] = rz;
            } else {
              r[0] = random->uniform() - 0.5;
              r[1] = random->uniform() - 0.5;
              r[2] = random->uniform() - 0.5;
            }
          } else {
            r[0] = r[1] = 0.0;
            r[2] = 1.0;
          }
          double theta = random->uniform() * MY_2PI;
          MathExtra::norm3(r);
          MathExtra::axisangle_to_quat(r,theta,quat);
          MathExtra::quat_to_mat(quat,rotmat);
          for (i = 0; i < natom; i++) {
            MathExtra::matvec(rotmat,onemols[imol]->dx[i],coords[i]);
            coords[i][0] += coord[0];
            coords[i][1] += coord[1];
            coords[i][2] += coord[2];

            imageflags[i] = ((imageint) IMGMAX << IMG2BITS) |
              ((imageint) IMGMAX << IMGBITS) | IMGMAX;
            domain->remap(coords[i],imageflags[i]);
          }
        }

        // check distance between any existing atom and any inserted atom
        // if less than near, try again
        // use minimum_image() to account for PBC

        flag = 0;
        for (m = 0; m < natom; m++) {
          for (i = 0; i < nlocal; i++) {
            delx = coords[m][0] - x[i][0];
            dely = coords[m][1] - x[i][1];
            delz = coords[m][2] - x[i][2];
            domain->minimum_image(delx,dely,delz);
            rsq = delx*delx + dely*dely + delz*delz;
            if (rsq < nearsq) flag = 1;
          }
        }
        // if (flagall) continue;

        // proceed with insertion

        //nlocalprev = atom->nlocal; not needed?

        // choose random velocity for new particle
        // used for every atom in molecule

        vnew[0] = vxlo + random->uniform() * (vxhi-vxlo);
        vnew[1] = vylo + random->uniform() * (vyhi-vylo);
        vnew[2] = vzlo + random->uniform() * (vzhi-vzlo);

        // if target specified, change velocity vector accordingly

        if (targetflag) {
          double vel = sqrt(vnew[0]*vnew[0] + vnew[1]*vnew[1] + vnew[2]*vnew[2]);
          delx = tx - coord[0];
          dely = ty - coord[1];
          delz = tz - coord[2];
          double rsq = delx*delx + dely*dely + delz*delz;
          if (rsq > 0.0) {
            double rinv = sqrt(1.0/rsq);
            vnew[0] = delx*rinv*vel;
            vnew[1] = dely*rinv*vel;
            vnew[2] = delz*rinv*vel;
          }
        }
      }
      if (flag_prob){
        added++;
        // out->coords has been allocated at the beginning with one element
        // if the number of added particles is 1 the new coords will be placed in
        // the array. Otherwise the array will be reallocated.
        if (added==1){
          out->coord[0] = new double[3];
          out->vel[0] = new double[3];
          for(int uu = 0; uu < 3; uu++){
            out->coord[0][uu] = coords[0][uu];
            out->vel[0][uu] = vnew[uu];
          }
        }else if (added > 1) {
        double** pt_coord = (double**) realloc(out->coord,1 + added * sizeof(out->coord[0]));
        double** pt_vel = (double**) realloc(out->vel, 1 + added * sizeof(out->vel[0]));
          out->coord = pt_coord;
          out->vel = pt_vel;
          out->coord[added-1] = new double[3];
          out->vel[added-1] = new double[3];
          for(int uu = 0; uu < 3; uu++){
            out->coord[added-1][uu] = coords[0][uu];
            out->vel[added-1][uu] = vnew[uu];
          }
        }
      }
    }
  }
  out->added = added;
}

void FixDuplicate::add_particles(struct added_coord in){
  /*
   * nfreq = total to add per step
   * ninsert = total amount of particles to add in whole simulation
   * nlocal = particles of current processor
   * ninserted = inserted particles
   */
  double *newcoord;
  int dimension = domain->dimension;
  int n, imol, nlocalprev;
  int flag_prob, flag, natom = 1;
  double lamda[3], vnew[3], coord[3], quat[4];
  double *sublo,*subhi;
  int added_all = in.added;
  int success = 1;
  int nlocal = atom->nlocal;

      if (domain->triclinic == 0) {
        sublo = domain->sublo;
        subhi = domain->subhi;
      } else {
        sublo = domain->sublo_lamda;
        subhi = domain->subhi_lamda;
      }
      for (int w = 0; w < added_all; w++){
        copy(&in.coord[w][0], &in.coord[w][2], &coords[0][0]);
        copy(&in.vel[w][0], &in.vel[w][2], &vnew[0]);

        // check if new atoms are in my sub-box or above it if I am highest proc
        // if so, add atom to my list via create_atom()
        // initialize additional info about the atoms
        // set group mask to "all" plus fix group
        if (random->uniform() < prob) flag_prob = 1;
        for (int m = 0; m < natom; m++) {
          if (domain->triclinic) {
            // domain->x2lamda(coords[m],lamda);
            domain->x2lamda(newcoord,lamda);
            newcoord = lamda;
          }else{
            for (int u = 0; u < 3; u++){
              newcoord[u] = in.coord[w][u];
            }
          }

          if (newcoord[0] >= sublo[0] && newcoord[0] < subhi[0] &&
              newcoord[1] >= sublo[1] && newcoord[1] < subhi[1] &&
              newcoord[2] >= sublo[2] && newcoord[2] < subhi[2]) flag = 1;
          else if (dimension == 3 && newcoord[2] >= domain->boxhi[2]) {
            if (comm->layout != Comm::LAYOUT_TILED) {
              if (comm->myloc[2] == comm->procgrid[2]-1 &&
                  newcoord[0] >= sublo[0] && newcoord[0] < subhi[0] &&
                  newcoord[1] >= sublo[1] && newcoord[1] < subhi[1]) flag = 1;
            } else {
              if (comm->mysplit[2][1] == 1.0 &&
                  newcoord[0] >= sublo[0] && newcoord[0] < subhi[0] &&
                  newcoord[1] >= sublo[1] && newcoord[1] < subhi[1]) flag = 1;
            }
          } else if (dimension == 2 && newcoord[1] >= domain->boxhi[1]) {
            if (comm->layout != Comm::LAYOUT_TILED) {
              if (comm->myloc[1] == comm->procgrid[1]-1 &&
                  newcoord[0] >= sublo[0] && newcoord[0] < subhi[0]) flag = 1;
            } else {
              if (comm->mysplit[1][1] == 1.0 &&
                  newcoord[0] >= sublo[0] && newcoord[0] < subhi[0]) flag = 1;
            }
          }
          // flag_prob=1;
          if (flag && flag_prob) {
            // to change in case of molecule?
            // if (mode == ATOM) atom->avec->create_atom(ntype,coords[m]);

            if (!domain->ownatom(maxmol_all + m + 1, newcoord, nullptr, 1)) continue;
            if (mode == ATOM) atom->avec->create_atom(group_child,newcoord);
            // else atom->avec->create_atom(ntype+onemols[imol]->type[m],coords[m]);
            else atom->avec->create_atom(ntype+onemols[imol]->type[m],newcoord);
            n = atom->nlocal - 1;
            atom->tag[n] = maxtag_all + m+1;
            if (mode == MOLECULE) {
              if (atom->molecule_flag) {
                if (onemols[imol]->moleculeflag) {
                  atom->molecule[n] = maxmol_all + onemols[imol]->molecule[m];
                } else {
                  atom->molecule[n] = maxmol_all+1;
                }
              }
              if (atom->molecular == Atom::TEMPLATE) {
                atom->molindex[n] = 0;
                atom->molatom[n] = m;
              }
            }
            // set mask of source atom
            atom->mask[n] = 1 | groupbit_child;
            atom->image[n] = imageflags[m];
            atom->v[n][0] = vnew[0];
            atom->v[n][1] = vnew[1];
            atom->v[n][2] = vnew[2];
            if (mode == MOLECULE) {
              onemols[imol]->quat_external = quat;
              atom->add_molecule_atom(onemols[imol],m,n,maxtag_all);
            }
            modify->create_attribute(n);
          }
        }

        // FixRigidSmall::set_molecule stores rigid body attributes
        //   coord is new position of geometric center of mol, not COM
        // FixShake::set_molecule stores shake info for molecule
        // nlocalprev = atom->nlocal;
        if (mode == MOLECULE) {
          if (rigidflag)
            fixrigid->set_molecule(nlocalprev,maxtag_all,imol,coord,vnew,quat);
          else if (shakeflag)
            fixshake->set_molecule(nlocalprev,maxtag_all,imol,coord,vnew,quat);
        }

        success = 1;
        // break;
      // }

      // warn if not successful b/c too many attempts

      if (!success && comm->me == 0)
        error->warning(FLERR,"Particle deposition was unsuccessful");

      // reset global natoms,nbonds,etc
      // increment maxtag_all and maxmol_all if necessary
      // if global map exists, reset it now instead of waiting for comm
      //   since other pre-exchange fixes may use it
      //   invoke map_init() b/c atom count has grown

      if (success && flag_prob) {
        atom->natoms += natom;
        if (atom->natoms < 0)
          error->all(FLERR,"Too many total atoms");
        if (mode == MOLECULE) {
          atom->nbonds += onemols[imol]->nbonds;
          atom->nangles += onemols[imol]->nangles;
          atom->ndihedrals += onemols[imol]->ndihedrals;
          atom->nimpropers += onemols[imol]->nimpropers;
        }
        maxtag_all += natom;
        if (maxtag_all >= MAXTAGINT)
          error->all(FLERR,"New atom IDs exceed maximum allowed ID");
        if (mode == MOLECULE && atom->molecule_flag) {
          if (onemols[imol]->moleculeflag) {
            maxmol_all += onemols[imol]->nmolecules;
          } else {
            maxmol_all++;
          }
        }
      }

      // rebuild atom map
      if (atom->map_style != Atom::MAP_NONE) {
        if (success && flag_prob) atom->map_init();
        atom->map_set();
      }
    }
  // next timestep to insert
  // next_reneighbor = 0 if done

  if (ninserted < ninsert) next_reneighbor += nfreq;
  else next_reneighbor = 0;
  if (success && flag_prob) ninserted++;
  // clean temporary array
  delete [] in.coord;
  delete [] in.vel;
}
void FixDuplicate::pre_exchange()
{
  /* MODE = ATOM = 0
     MODE = MOLECULE = 1*/
  struct added_coord out;
  // each processor will generate the position of the new particles 
  generate_positions(&out);
  // get rank and total rank
  int rank;
  MPI_Comm_rank(world, &rank);
  int max_rank;
  MPI_Comm_size(world, &max_rank);
  int local_added = out.added * 3;
  int global_added = 0;
  int nlocal = atom->nlocal;
  MPI_Allreduce(&out.added, &global_added, 1, MPI_INT, MPI_SUM, world);

  int counts[max_rank];
  if (!idnext) find_maxid();
  MPI_Allgather(&local_added, 1, MPI_INT, counts, 1, MPI_INT, world);
  // create count data for MPI_Allgatherv
  int count_sum = 0;
  for (int i = 0; i < max_rank ; i++){
    count_sum += counts[i];
  }
  // create displacements data for MPI_Allgatherv
  int displs[max_rank] = {0};
  displs[0] = 0;
  for (int i = 0; i < max_rank -1 ; i++){
    displs[i+1] = displs[i] + counts[i];
  }

  // create a flat the local_coords array to sendit with MPI;
  // place all local coordinate in a flat array (assuming that each coordinate has x,y,z values)
  double *local_coords_flat = new double[count_sum];
  for (int i = 0; i < out.added; i++){
    for(int j = 0; j < 3; j++){
      local_coords_flat[i*3+j] = out.coord[i][j];
    }
  }
  double *local_vel_flat = new double[count_sum];
  for (int i = 0; i < out.added; i++){
    for(int j = 0; j < 3; j++){
      local_vel_flat[i*3+j] = out.vel[i][j];
    }
  }

  // create flat global buffer
  double *global_coords_buffer = new double[count_sum];
  double *global_vel_buffer = new double[count_sum];
  // gather array (the same for all processors) that contains all particles coordinates previously generated
  MPI_Allgatherv(local_coords_flat, local_added, MPI_DOUBLE, global_coords_buffer, counts, displs, MPI_DOUBLE, world);
  MPI_Allgatherv(local_vel_flat, local_added, MPI_DOUBLE, global_vel_buffer, counts, displs, MPI_DOUBLE, world);

  // rebuild global_coords adding all coordinates in the global array
  double **global_coords = new double*[global_added];
  for (int i = 0; i < global_added; i++){
    global_coords[i] = new double[3];
    for (int j = 0; j < 3; j++){
      global_coords[i][j] = global_coords_buffer[i*3+j];
    }
  }
  double **global_vel = new double*[global_added];
  for (int i = 0; i < global_added; i++){
    global_vel[i] = new double[3];
    for (int j = 0; j < 3; j++){
      global_vel[i][j] = global_vel_buffer[i*3+j];
    }
  }
  struct added_coord global_out;
  global_out.added = global_added;
  global_out.coord = global_coords;
  global_out.vel = global_vel;
  // add the particles to the system
  add_particles(global_out);
}

/* ----------------------------------------------------------------------
   maxtag_all = current max atom ID for all atoms
   maxmol_all = current max molecule ID for all atoms
------------------------------------------------------------------------- */

void FixDuplicate::find_maxid()
{
  tagint *tag = atom->tag;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;

  tagint max = 0;
  for (int i = 0; i < nlocal; i++) max = MAX(max,tag[i]);
  MPI_Allreduce(&max,&maxtag_all,1,MPI_LMP_TAGINT,MPI_MAX,world);

  if (mode == MOLECULE && molecule) {
    max = 0;
    for (int i = 0; i < nlocal; i++) max = MAX(max,molecule[i]);
    MPI_Allreduce(&max,&maxmol_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
  }
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */

void FixDuplicate::options(int narg, char **arg)
{
  // defaults
  iregion = nullptr;
  idregion = nullptr;
  mode = ATOM;
  molfrac = nullptr;
  rigidflag = 0;
  idrigid = nullptr;
  shakeflag = 0;
  idshake = nullptr;
  idnext = 0;
  globalflag = localflag = 0;
  lo = hi = deltasq = 0.0;
  nearsq = 0.0;
  maxattempt = 10;
  rateflag = 0;
  vxlo = vxhi = vylo = vyhi = vzlo = vzhi = 0.0;
  distflag = DIST_UNIFORM;
  sigma = 1.0;
  xmid = ymid = zmid = 0.0;
  scaleflag = 1;
  targetflag = 0;
  orientflag = 0;
  rx = 0.0;
  ry = 0.0;
  rz = 0.0;

  int iarg = 0;
  while (iarg < narg) {
    // if (strcmp(arg[iarg],"region") == 0) {
      // if (iarg+2 > narg) error->all(FLERR,"Illegal fix deposit command");
      // iregion = domain->get_region_by_id(arg[iarg+1]);
      // if (!iregion) error->all(FLERR,"Region ID {} for fix deposit does not exist",arg[iarg+1]);
      // idregion = utils::strdup(arg[iarg+1]);
      // iarg += 2;
    // } else if (strcmp(arg[iarg],"mol") == 0) {
    if(strcmp(arg[iarg],"mol") == 0){
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix deposit command");
      int imol = atom->find_molecule(arg[iarg+1]);
      if (imol == -1) error->all(FLERR,"Molecule template ID for fix deposit does not exist");
      mode = MOLECULE;
      onemols = &atom->molecules[imol];
      nmol = onemols[0]->nset;
      delete [] molfrac;
      molfrac = new double[nmol];
      molfrac[0] = 1.0/nmol;
      for (int i = 1; i < nmol-1; i++) molfrac[i] = molfrac[i-1] + 1.0/nmol;
      molfrac[nmol-1] = 1.0;
      iarg += 2;
    } else if (strcmp(arg[iarg],"molfrac") == 0) {
      if (mode != MOLECULE) error->all(FLERR,"Illegal fix deposit command");
      if (iarg+nmol+1 > narg) error->all(FLERR,"Illegal fix deposit command");
      molfrac[0] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      for (int i = 1; i < nmol; i++)
        molfrac[i] = molfrac[i-1] + utils::numeric(FLERR,arg[iarg+i+1],false,lmp);
      if (molfrac[nmol-1] < 1.0-EPSILON || molfrac[nmol-1] > 1.0+EPSILON)
        error->all(FLERR,"Illegal fix deposit command");
      molfrac[nmol-1] = 1.0;
      iarg += nmol+1;
    } else if (strcmp(arg[iarg],"rigid") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix deposit command");
      delete [] idrigid;
      idrigid = utils::strdup(arg[iarg+1]);
      rigidflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"shake") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix deposit command");
      delete [] idshake;
      idshake = utils::strdup(arg[iarg+1]);
      shakeflag = 1;
      iarg += 2;

    } else if (strcmp(arg[iarg],"id") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix deposit command");
      if (strcmp(arg[iarg+1],"max") == 0) idnext = 0;
      else if (strcmp(arg[iarg+1],"next") == 0) idnext = 1;
      else error->all(FLERR,"Illegal fix deposit command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"global") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix deposit command");
      globalflag = 1;
      localflag = 0;
      lo = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      hi = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      iarg += 3;
    } else if (strcmp(arg[iarg],"local") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix deposit command");
      localflag = 1;
      globalflag = 0;
      lo = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      hi = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      deltasq = utils::numeric(FLERR,arg[iarg+3],false,lmp) *
        utils::numeric(FLERR,arg[iarg+3],false,lmp);
      iarg += 4;

    } else if (strcmp(arg[iarg],"near") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix deposit command");
      nearsq = utils::numeric(FLERR,arg[iarg+1],false,lmp) *
        utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"attempt") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix deposit command");
      maxattempt = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"rate") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix deposit command");
      rateflag = 1;
      rate = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"vx") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix deposit command");
      vxlo = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      vxhi = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      iarg += 3;
    } else if (strcmp(arg[iarg],"vy") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix deposit command");
      vylo = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      vyhi = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      iarg += 3;
    } else if (strcmp(arg[iarg],"vz") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix deposit command");
      vzlo = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      vzhi = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      iarg += 3;
    } else if (strcmp(arg[iarg],"orient") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix deposit command");
      orientflag = 1;
      rx = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      ry = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      rz = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      if (domain->dimension == 2 && (rx != 0.0 || ry != 0.0))
        error->all(FLERR,"Illegal fix deposit orient settings");
      if (rx == 0.0 && ry == 0.0 && rz == 0.0)
        error->all(FLERR,"Illegal fix deposit orient settings");
      iarg += 4;
    } else if (strcmp(arg[iarg],"units") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix deposit command");
      if (strcmp(arg[iarg+1],"box") == 0) scaleflag = 0;
      else if (strcmp(arg[iarg+1],"lattice") == 0) scaleflag = 1;
      else error->all(FLERR,"Illegal fix deposit command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"gaussian") == 0) {
      if (iarg+5 > narg) error->all(FLERR,"Illegal fix deposit command");
      xmid = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      ymid = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      zmid = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      sigma = utils::numeric(FLERR,arg[iarg+4],false,lmp);
      distflag = DIST_GAUSSIAN;
      iarg += 5;
    } else if (strcmp(arg[iarg],"target") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix deposit command");
      tx = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      ty = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      tz = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      targetflag = 1;
      iarg += 4;
    } else error->all(FLERR,"Illegal fix deposit command");
  }
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixDuplicate::write_restart(FILE *fp)
{
  int n = 0;
  double list[5];
  list[n++] = random->state();
  list[n++] = ninserted;
  list[n++] = ubuf(nfirst).d;
  list[n++] = ubuf(next_reneighbor).d;
  list[n++] = ubuf(update->ntimestep).d;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixDuplicate::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;

  seed = static_cast<int>(list[n++]);
  ninserted = static_cast<int>(list[n++]);
  nfirst = static_cast<bigint>(ubuf(list[n++]).i);
  next_reneighbor = static_cast<bigint>(ubuf(list[n++]).i);

  bigint ntimestep_restart = static_cast<bigint>(ubuf(list[n++]).i);
  if (ntimestep_restart != update->ntimestep)
    error->all(FLERR,"Must not reset timestep when restarting this fix");

  random->reset(seed);
}

/* ----------------------------------------------------------------------
   extract particle radius for atom type = itype
------------------------------------------------------------------------- */

void *FixDuplicate::extract(const char *str, int &itype)
{
  if (strcmp(str,"radius") == 0) {
    if (mode == ATOM) {
      if (itype == ntype) oneradius = 0.5;
      else oneradius = 0.0;

    } else {

      // loop over onemols molecules
      // skip a molecule with no atoms as large as itype

      oneradius = 0.0;
      for (int i = 0; i < nmol; i++) {
        if (itype > ntype+onemols[i]->ntypes) continue;
        double *radius = onemols[i]->radius;
        int *type = onemols[i]->type;
        int natoms = onemols[i]->natoms;

        // check radii of atoms in Molecule with matching types
        // default to 0.5, if radii not defined in Molecule
        //   same as atom->avec->create_atom(), invoked in pre_exchange()

        for (int i = 0; i < natoms; i++)
          if (type[i]+ntype == itype) {
            if (radius) oneradius = MAX(oneradius,radius[i]);
            else oneradius = MAX(oneradius,0.5);
          }
      }
    }
    itype = 0;
    return &oneradius;
  }

  return nullptr;
}

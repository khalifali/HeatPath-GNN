#!/usr/bin/env python3
"""Independent contact-model audit.

Run: python3 test_contact_model.py
Known discrepancies are ordinary failures, not hidden by expectedFailure.
Reference: LAMMPS stable_22Jul2025_update5, pair_granular.rst.
For a circular isothermal contact, each half-space contributes 1/(4*k*a).
This tests that ideal constriction model, not experimental bed validity.
"""
import math
from pathlib import Path
import re
import unittest
import numpy as np
import solve_packing_heat_transfer as s


def reference_g(k1, k2, a):
    # Two spreading resistances in series; independent of production helpers.
    return 1.0 / (1.0 / (4*k1*a) + 1.0 / (4*k2*a))


class ContactModelTests(unittest.TestCase):
    def close(self, actual, expected):
        self.assertTrue(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-15),
                        f"actual={actual:.12g}, reference={expected:.12g}")

    def test_equal_material_absolute_conductance(self):
        self.close(s.harmonic_contact_conductance(2., 2., 1e-4), 4e-4)

    def test_dissimilar_material_series_resistance(self):
        self.close(s.harmonic_contact_conductance(1., 10., 1e-4),
                   reference_g(1., 10., 1e-4))

    def test_finite_wall_series_resistance(self):
        self.close(s.wall_contact_conductance(1., 10., 1e-4),
                   reference_g(1., 10., 1e-4))

    def test_isothermal_wall_limit(self):
        self.close(s.wall_contact_conductance(2., math.inf, 1e-4), 8e-4)

    def test_zero_radius_and_exchange_symmetry(self):
        self.close(s.harmonic_contact_conductance(1., 10., 0.), 0.)
        self.close(s.harmonic_contact_conductance(1., 10., 1e-4),
                   s.harmonic_contact_conductance(10., 1., 1e-4))

    def test_hertz_radius_from_independent_force_relation(self):
        radius1, radius2, overlap, modulus = 1e-3, 1.5e-3, 1e-6, 3e7
        effective_radius = 1/(1/radius1 + 1/radius2)
        force = (4/3)*modulus*math.sqrt(effective_radius)*overlap**1.5
        expected_radius = (3*force*effective_radius/(4*modulus))**(1/3)
        particles = [s.Particle(1,1,0,0,0,radius1,1),
                     s.Particle(2,1,0,0,radius1+radius2-overlap,radius2,1)]
        box = s.Box(np.zeros(3), np.ones(3))
        edges, _ = s.build_contact_edges(particles,box,[(1,2)],np.ones(2),0.)
        self.close(edges[0].contact_radius,expected_radius)

    def test_periodic_contact_radius(self):
        # Separation is 0.00199 m across the periodic x boundary.
        particles = [s.Particle(1,1,0.000995,0.005,0.005,0.001,1),
                     s.Particle(2,1,0.009005,0.005,0.005,0.001,1)]
        box = s.Box(np.zeros(3),np.full(3,0.01))
        edges,_ = s.build_contact_edges(particles,box,[(1,2)],np.ones(2),0.)
        self.close(edges[0].contact_radius,math.sqrt(0.0005*0.00001))

    def test_two_particle_column_absolute_heat_rate(self):
        radius, delta, k = 1e-3, 1e-6, 2.
        height = 4*radius-3*delta
        particles = [s.Particle(1,1,0.005,0.005,radius-delta,radius,1),
                     s.Particle(2,1,0.005,0.005,3*radius-2*delta,radius,1)]
        box = s.Box(np.zeros(3),np.array([0.01,0.01,height]))
        edges,_ = s.build_contact_edges(particles,box,[(1,2)],np.full(2,k),0.)
        hot,cold,_,_ = s.build_wall_links(
            particles,box,np.full(2,k),math.inf,0.,0.,0.,0.,0.)
        _,_,qhot,qcold,error,_ = s.solve_network(
            2,edges,hot,cold,1.,0.,np.ones(2,dtype=bool))
        # Two wall spreading resistances plus the two-sided interior contact.
        awall = math.sqrt(radius*delta)
        apair = math.sqrt(radius*delta/2)
        expected_q = 1/(2/(4*k*awall)+2/(4*k*apair))
        self.assertLess(error,1e-10)
        self.close(qhot,qcold)
        self.close(qhot,expected_q)

    def test_dem_material_argument_order(self):
        source = Path(__file__).with_name("in.generate_equal_porosity_psd.lammps").read_text()
        source = "\n".join(line.split("#",1)[0] for line in source.splitlines())
        orders = re.findall(r"hertz/material\s+(\S+)\s+(\S+)\s+(\S+)",source)
        self.assertEqual(len(orders),2)  # particle and wall definitions
        for order in orders:
            self.assertEqual(order,("${young}","${restitution}","${poisson}"),
                             "LAMMPS expects E, restitution (Tsuji), Poisson ratio")


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""

A particle class

"""
Vector = list[float]

class Particle:

    def __init__(self, mass:float, initPosx:float, initPosy:float, 
                                   initVelx:float, initVely:float):
        """
        

        :param mass: the mass of the particle
        :param initPosx: the initial x position
        :param initPosy: the initial y position
        :param initVelx: the initial x velocity
        :param initVely: the initial y velocity

        """
        self.mass = mass
        self.posx = initPosx
        self.posy = initPosy
        self.velx = initVelx
        self.vely = initVely

        return

    def get_mass(self) -> float:

        """
        
        :return: get the particle mass
        """
        return self.mass

    def get_velocity(self) -> Vector:
        """
        Get the particle velocity [vx, vy]

        :return: the particle velocity in [vx, vy]
        """
        return [self.velx, self.vely]

    def apply_forces(self, force:Vector):
        """
        add a force to this particle

        :param force: the force in [fx, fy]
        """
        self.fx = force[0]
        self.fy = force[1]
        return

    def get_force(self) -> Vector:
        """
        get the force [fx, fy]

        :return: the force applied on this particle [fx, fy]
        """
        return [self.fx, self.fy]

    def get_accerlation(self) -> Vector:
        """
        get the accerlation of this particle

        :return: the accerlation of this particle
        """
        mass = self.mass
        return [self.fx/mass, self.fy/mass]

if __name__=='__main__':


    particle = Particle(mass=1.0, 
                        initPosx=0.0, 
                        initPosy=0.0, 
                        initVelx=1.0, 
                        initVely=1.0)

    mass = particle.get_mass()
    print("Mass = ", mass)
    force = [0.0, -9.8]
    particle.apply_forces(force)
    acc = particle.get_accerlation()
    print("Acc =", acc)
    print("Done")
import pygame, numpy, glm

windowsize = [128, 128]
camdata = [[0, 0, 0], [0, 0, 0]]
maxiterations = 500
maxdist = 500
mindist = 0.01
EPSILON = 0.0001

def intersectSDF(distA, distB):
    return glm.max(distA, distB)

def unionSDF(distA, distB):
    return glm.min(distA, distB)

def differenceSDF(distA, distB):
    return glm.max(distA, -distB)

def boxSDF(p, size):
    d = glm.abs(p)-(size/2.0)
    insideDistance = glm.min(glm.max(d.x, glm.max(d.y, d.z)), 0.0)
    outsideDistance = glm.length(glm.max(d, 0.0))
    return insideDistance+outsideDistance

def sphereSDF(p, r):
    return glm.length(p)-r

def cylinderSDF(p, h, r):
    inOutRadius = glm.length(p.xy)-r
    inOutHeight = glm.abs(p.z)-h/2.0
    insideDistance = glm.min(glm.max(inOutRadius, inOutHeight), 0.0)
    outsideDistance = glm.length(glm.max(glm.vec2(inOutRadius, inOutHeight), 0.0))
    return insideDistance+outsideDistance

def rotationMatrix3(v, angle):
    c = numpy.cos(numpy.radians(angle))
    s = numpy.sin(numpy.radians(angle))
    return glm.mat3(c+(1-c)*v.x*v.x, (1-c)*v.x*v.y-s*v.z, (1-c)*v.x*v.z+s*v.y,
                    (1-c)*v.x*v.y+s*v.z, c+(1-c)*v.y*v.y, (1-c)*v.y*v.z-s*v.x,
                    (1-c)*v.x*v.z-s*v.y, (1-c)*v.y*v.z+s*v.x, c+(1-c)*v.z*v.z)

def powN1(z, r, dr, power):
    theta = numpy.arccos(z.z/r)
    phi = numpy.arctan2(z.y, z.x)
    dr = pow(r, power-1)*power*dr+1
    zr = pow(r, power)
    theta = theta*power
    phi = phi*power
    z = zr*glm.vec3(numpy.sin(theta)*numpy.cos(phi), numpy.sin(phi)*numpy.sin(theta), numpy.cos(theta))
    return [z, dr]
    
def powN2(z, zr0, dr, power):
    zo0 = numpy.arcsin(z.z/zr0)
    zi0 = numpy.arctan2(z.y, z.x)
    zr = pow(zr0, power-1)
    zo = zo0*power
    zi = zi0*power
    dr = zr*dr*power+1
    zr *= zr0
    z = zr*glm.vec3(numpy.cos(zo)*numpy.cos(zi), numpy.cos(zo)*numpy.sin(zi), numpy.sin(zo))
    return [z, dr]

def mandelbulbSDF(pos, iterations, power):
    julia = False
    juliaC = glm.vec3(0, 0, 0) #(-2, -2, -2) to (2, 2, 2)
    orbitTrap = glm.vec4(10000)
    colorIterations = 9 #0 to 100
    bailout = 5 #0 to 30
    alternateVersion = False
    rotVector = glm.vec3(1, 1, 1) #(0, 0, 0) to (1, 1, 1)
    rotAngle = 0 #0 to 180(?)
    rot = rotationMatrix3(glm.normalize(rotVector), rotAngle)
    z = pos
    dr = 1
    i = 0
    r = glm.length(z)
    while r < bailout and i < iterations:
        if alternateVersion:
            z, dr = powN2(z, r, dr, power)
        else:
            z, dr = powN1(z, r, dr, power)
        z += juliaC if julia else pos
        r = glm.length(z)
        z *= rot
        if i < colorIterations:
            orbitTrap = min(orbitTrap, abs(glm.vec4(z.x, z.y, z.z, r*r)))
        i += 1
    return 0.5*numpy.log(r)*r/dr

def mandelboxSDF(pos, scale):
    orbitTrap = glm.vec4(10000)
    iterations = 17 #0 to 300
    colorIterations = 3 #0 to 300
    minRad2 = 0.25 #0.0 to 2.0
    scale = glm.vec4(scale, scale, scale, abs(scale))/minRad2
    rotVector = glm.vec3(1, 1, 1) #(0, 0, 0) to (1, 1, 1)
    rotAngle = 0 #0 to 180(?)
    rot = rotationMatrix3(glm.normalize(rotVector), rotAngle)
    absScalem1 = abs(scale-1)
    absScaleRaisedTo1mIters = pow(abs(scale), float(1-iterations))
    p = pos
    w = 1
    p0 = p
    w0 = w
    for i in range(iterations):
        p *= rot
        p = glm.clamp(p, -1, 1)*2-p
        r2 = glm.dot(p, p)
        if i < colorIterations:
            orbitTrap = glm.min(orbitTrap, abs(glm.vec4(p, r2)))
        p *= glm.clamp(glm.max(minRad2/r2, minRad2), 0, 1)
        w *= glm.clamp(glm.max(minRad2/r2, minRad2), 0, 1)
        p = p*glm.vec3(scale[0], scale[1], scale[2])+p0
        w = w*scale[3]+w0
        if r2 > 1000:
            break
    return (glm.length(p)-absScalem1[3])/w-absScaleRaisedTo1mIters[3]

def sceneSDF2(p):
    #cube = boxSDF(p, glm.vec3(1.8, 1.8, 1.8))
    #sphere = sphereSDF(p, 1.2)
    #return intersectSDF(cube, sphere)
    return mandelboxSDF(p, 2.5)
    #return mandelbulbSDF(p, 9, 8)

def estimateNormal(p):
    return glm.normalize(glm.vec3(
        sceneSDF2(glm.vec3(p[0]+EPSILON, p[1], p[2]))-sceneSDF2(glm.vec3(p[0]-EPSILON, p[1], p[2])),
        sceneSDF2(glm.vec3(p[0], p[1]+EPSILON, p[2]))-sceneSDF2(glm.vec3(p[0], p[1]-EPSILON, p[2])),
        sceneSDF2(glm.vec3(p[0], p[1], p[2]+EPSILON))-sceneSDF2(glm.vec3(p[0], p[1], p[2]-EPSILON))
    ))

def get_pixel_color(pixel_coords):
    tde = 0
    objpos = glm.vec3(0, 0, 10)
    campos = glm.vec3(camdata[0])
    pos = campos
    rot = glm.vec3(camdata[1][0]+numpy.arctan2((pixel_coords[1]-windowsize[1]/2)/windowsize[1], 0.5),
                   camdata[1][1]+numpy.arctan2((pixel_coords[0]-windowsize[0]/2)/windowsize[1], 0.5),
                   camdata[1][2])
    direction = glm.vec3(numpy.sin(rot[1])*numpy.cos(rot[0]),
                         numpy.sin(rot[0]),
                         numpy.cos(rot[1])*numpy.cos(rot[0]))
    for i in range(maxiterations):
        pos += direction*tde
        tde = sceneSDF2(objpos-pos)
        if tde < mindist:
            estNrm = estimateNormal(objpos-pos)
            return estNrm*127+127
        elif glm.length(pos-campos) > maxdist:
            break
    return (63, 63, 63)

pygame.init()
surface = pygame.display.set_mode(windowsize)
pygame.display.set_caption("Ray Marching Thing")
clock = pygame.time.Clock()
running = True
while running:
    clock.tick(60)
    for x in range(windowsize[0]):
        for y in range(windowsize[1]):
            pixelcolor = get_pixel_color([x, y])
            surface.set_at([x, y], pixelcolor)
    pygame.display.flip()
pygame.quit()

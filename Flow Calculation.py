

def calculate_reynolds_number(freestream_density, freestream_velocity, reference_length, freestream_viscosity):
    return freestream_density * freestream_velocity * reference_length / freestream_viscosity

def calculate_flow_type(reynolds_number):
    if reynolds_number < 5e5:
        return "laminar"
    elif 5e5 <= reynolds_number <= 5e5:  # Missing condition here
        return "transition"
    else:
        return "turbulent"

def calculate_boundary_layer_thickness(reynolds_number, x):
    if reynolds_number <= 5e5:
        return 5.2 * x / reynolds_number**0.5
    else:
        return 0.37 * x / reynolds_number**0.2

def calculate_local_skin_friction_coefficient(reynolds_number):
    if reynolds_number <= 5e5:
        return 0.664 / reynolds_number**0.5
    else:
        return 0.0592 / reynolds_number**0.2
    
def calculate_total_skin_friction_drag_coefficient(reynolds_number):
    if reynolds_number <= 5e5:
        return 1.328 / reynolds_number**0.5
    else:
        return 0.074 / reynolds_number**0.2

def calculate_ke_of_flow(freestream_density, freestream_velocity):
    return 0.5 * freestream_density * freestream_velocity**2

def calculate_reference_area(reference_length, reference_width):
    return reference_length * reference_width 

def calculate_total_skin_friction_drag(total_skin_friction_drag_coefficient, reference_area, ke_of_flow):
    return total_skin_friction_drag_coefficient * ke_of_flow * reference_area

def calculate_shear_stress_wall(reynolds_number_1):
    return 0.664 / reynolds_number_1**0.5

# Input parameters
freestream_velocity = 10.0  # m/s
reference_length_1 = 0.4  # m
x = reference_length_1
reference_length = 1 # m
reference_width = 0.2 # m
freestream_density = 1 # kg/m**3
freestream_viscosity = 1e-5  # You need to provide the viscosity value

reynolds_number_1 = calculate_reynolds_number(freestream_density, freestream_velocity, reference_length_1, freestream_viscosity)

reynolds_number = calculate_reynolds_number(freestream_density, freestream_velocity, reference_length, freestream_viscosity)

flow_type = calculate_flow_type(reynolds_number)

boundary_layer_thickness = calculate_boundary_layer_thickness(reynolds_number, reference_length)

local_skin_friction_coefficient = calculate_local_skin_friction_coefficient(reynolds_number)

total_skin_friction_drag_coefficient = calculate_total_skin_friction_drag_coefficient(reynolds_number)

ke_of_flow = calculate_ke_of_flow(freestream_density, freestream_velocity)

reference_area = calculate_reference_area(reference_length, reference_width)

total_skin_friction_drag = calculate_total_skin_friction_drag(total_skin_friction_drag_coefficient, reference_area, ke_of_flow)

shear_stress_wall = calculate_shear_stress_wall(reynolds_number_1)

print("For Incompressible flow:")
print("Flow type is:", flow_type)
print("Boundary Layer Thickness:", boundary_layer_thickness)
print("Local Skin Friction Coefficient:", local_skin_friction_coefficient)
print("Total Skin Friction Drag Coefficient:", total_skin_friction_drag_coefficient)
print("Total Skin Friction Drag:", total_skin_friction_drag)
print("Shear Stress on the wall is:", shear_stress_wall)

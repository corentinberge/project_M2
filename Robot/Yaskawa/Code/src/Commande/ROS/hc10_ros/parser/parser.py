import xml.dom.minidom as md

def modifyElement(file, typeTag, name, field, property, attribute, value):
    '''
    Modify a value of the property of th specified element

    Parameters
    ----------
    file: Document
        The file to modify
    typeTag: String
        Type of element (ex: 'joint', 'link')
    name: String
        Name of element (ex: 'joint1')
    field: String
        Field where the value is (ex: 'inertial', 'visual', 'collision')
        Can be set to None
    property: String
        Property to modify (ex: 'origin', 'mass', 'inertia' ...)
    attribute: String
        Attribute to modify (ex: 'xyz', 'rpy', 'value' ...)
    value: String
        Replacement value (ex: '1 0 1', '0' ...)
    '''
    elements = file.getElementsByTagName(typeTag)
    for e in elements:
        if e.getAttribute('name') == name:
            field = e
            if(field != None and e.getElementsByTagName(field)):
                field = e.getElementsByTagName(field)[0]
            prop = field.getElementsByTagName(property)[0]
            if (prop.getAttribute(attribute)):
                prop.setAttribute(attribute, value)


def main():
    filename = 'planar_2DOF.urdf'

    # Load urdf file
    file = md.parse(filename)

    # Type of file
    print('Type of file:', file.nodeName)

    # Process file
    modifyElement(file, 'link', 'base_link', 'inertial', 'origin', 'rpy', '1 1 1')
    modifyElement(file, 'joint', 'joint1', None, 'axis', 'xyz', '1 0 0')
    # ...

    # Save file
    print('Saving file ...')
    with open('modified_'+filename, 'w') as fs:
        fs.write(file.toxml())
        fs.close()
    print('Done!')


if __name__ == '__main__':
    main()

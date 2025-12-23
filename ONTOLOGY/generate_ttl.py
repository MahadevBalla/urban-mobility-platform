"""
Generate TTL (Turtle) RDF Ontology from Python Dataclasses

This script parses all ontology Python modules and creates a formal
RDF/OWL ontology in Turtle format by extracting:
- All class definitions
- All dataclass fields
- Field types and relationships
- Class inheritance hierarchies
- Enumerations

Only extracts what's actually defined in the code - no assumptions.
"""

import ast
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime

# Add ONTOLOGY to path
ONTOLOGY_DIR = Path(__file__).parent
sys.path.insert(0, str(ONTOLOGY_DIR.parent))

# Import all modules to extract runtime information
from ONTOLOGY import ontology_base
from ONTOLOGY import ontology_census
from ONTOLOGY import ontology_hts
from ONTOLOGY import ontology_mobile
from ONTOLOGY import ontology_probe
from ONTOLOGY import ontology_gtfs
from ONTOLOGY import ontology_osm


class OntologyExtractor:
    """Extract ontology structure from Python modules"""
    
    def __init__(self):
        self.classes = {}
        self.enums = {}
        self.dataclasses = {}
        self.relationships = []
        self.modules = [
            ('base', ontology_base),
            ('census', ontology_census),
            ('hts', ontology_hts),
            ('mobile', ontology_mobile),
            ('probe', ontology_probe),
            ('gtfs', ontology_gtfs),
            ('osm', ontology_osm)
        ]
        
    def extract_all(self):
        """Extract from all modules"""
        print("Extracting ontology from Python modules...")
        
        for module_name, module in self.modules:
            print(f"  Processing {module_name}...")
            self.extract_from_module(module_name, module)
        
        print(f"\nExtracted:")
        print(f"  - {len(self.enums)} enumerations")
        print(f"  - {len(self.dataclasses)} dataclasses")
        print(f"  - {len(self.relationships)} relationships")
    
    def _parse_source_for_enums(self, module_name: str, module_file: Path):
        """Parse source file to extract enumerations"""
        try:
            with open(module_file, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's an Enum subclass
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == 'Enum':
                            # Extract enum values
                            values = []
                            for item in node.body:
                                if isinstance(item, ast.Assign):
                                    for target in item.targets:
                                        if isinstance(target, ast.Name):
                                            values.append(target.id)
                            
                            if values and node.name not in self.enums:
                                self.enums[node.name] = {
                                    'module': module_name,
                                    'values': values
                                }
        except Exception as e:
            print(f"    Warning: Could not parse {module_file.name}: {e}")
        
    def extract_from_module(self, module_name: str, module):
        """Extract classes from a module"""
        # Also parse source code for enumerations
        module_file = Path(module.__file__)
        if module_file.exists():
            self._parse_source_for_enums(module_name, module_file)
        
        for name, obj in inspect.getmembers(module):
            # Skip private/imported
            if name.startswith('_') or inspect.getmodule(obj) != module:
                continue
                
            # Extract enumerations using inspect
            try:
                from enum import Enum
                if inspect.isclass(obj) and issubclass(obj, Enum):
                    if hasattr(obj, '__members__') and name not in self.enums:
                        self.enums[name] = {
                            'module': module_name,
                            'values': list(obj.__members__.keys())
                        }
            except:
                pass
            
            # Extract dataclasses
            if hasattr(obj, '__dataclass_fields__'):
                fields = {}
                for field_name, field_obj in obj.__dataclass_fields__.items():
                    field_type = self._get_type_string(field_obj.type)
                    default = field_obj.default if field_obj.default != field_obj.default_factory else None
                    
                    fields[field_name] = {
                        'type': field_type,
                        'required': default is None and field_obj.default_factory is None
                    }
                
                # Get parent classes
                parents = [base.__name__ for base in obj.__bases__ if base.__name__ != 'object']
                
                self.dataclasses[name] = {
                    'module': module_name,
                    'fields': fields,
                    'parents': parents,
                    'docstring': obj.__doc__
                }
                
                # Track relationships
                for field_name, field_info in fields.items():
                    if '_id' in field_name and field_name != 'entity_id':
                        # This is likely a foreign key relationship
                        target = field_name.replace('_id', '').replace('_', '')
                        self.relationships.append({
                            'source': name,
                            'target': target,
                            'property': field_name,
                            'type': 'reference'
                        })
    
    def _get_type_string(self, type_hint) -> str:
        """Convert Python type hint to string"""
        type_str = str(type_hint)
        
        # Clean up typing module references
        type_str = type_str.replace('typing.', '')
        type_str = type_str.replace('<class \'', '').replace('\'>', '')
        
        return type_str
    
    def generate_ttl(self, output_file: str):
        """Generate Turtle RDF ontology"""
        print(f"\nGenerating TTL ontology: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write(self._generate_header())
            
            # Write enumerations
            f.write(self._generate_enumerations())
            
            # Write classes
            f.write(self._generate_classes())
            
            # Write properties
            f.write(self._generate_properties())
            
            # Write relationships
            f.write(self._generate_relationships())
        
        print(f"✅ TTL ontology generated: {output_file}")
    
    def _generate_header(self) -> str:
        """Generate TTL header with prefixes"""
        return f"""# Transport Data Ontology
# Version: 1.0.0
# Generated: {datetime.now().isoformat()}
# Source: Python dataclass modules in ONTOLOGY/

@prefix : <http://transport-ontology.org/ontology#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> .
@prefix time: <http://www.w3.org/2006/time#> .

# Ontology Metadata
: a owl:Ontology ;
    dcterms:title "Transport Data Ontology" ;
    dcterms:description "Comprehensive ontology for urban transport data integration, aligned with international standards (Transmodel, GTFS, NHTS, Census)" ;
    dcterms:created "{datetime.now().date().isoformat()}"^^xsd:date ;
    dcterms:creator "Urban Transit Tool Project" ;
    owl:versionInfo "1.0.0" .

"""
    
    def _generate_enumerations(self) -> str:
        """Generate enumeration definitions"""
        ttl = "\n# ==================================================\n"
        ttl += "# ENUMERATIONS\n"
        ttl += "# ==================================================\n\n"
        
        for enum_name, enum_info in sorted(self.enums.items()):
            ttl += f"# Enumeration: {enum_name}\n"
            ttl += f":{enum_name} a rdfs:Class , owl:Class ;\n"
            ttl += f"    rdfs:label \"{enum_name}\" ;\n"
            ttl += f"    rdfs:subClassOf :Enumeration ;\n"
            ttl += f"    dcterms:source \"{enum_info['module']} module\" .\n\n"
            
            # Add enumeration values
            for value in enum_info['values']:
                ttl += f":{enum_name}_{value} a :{enum_name} ;\n"
                ttl += f"    rdfs:label \"{value}\" ;\n"
                ttl += f"    skos:notation \"{value}\" .\n\n"
        
        return ttl
    
    def _generate_classes(self) -> str:
        """Generate class definitions"""
        ttl = "\n# ==================================================\n"
        ttl += "# CLASSES\n"
        ttl += "# ==================================================\n\n"
        
        for class_name, class_info in sorted(self.dataclasses.items()):
            # Class definition
            ttl += f"# Class: {class_name}\n"
            if class_info['docstring']:
                docstring = class_info['docstring'].strip().replace('\n', ' ')
                ttl += f"# {docstring}\n"
            
            ttl += f":{class_name} a rdfs:Class , owl:Class ;\n"
            ttl += f"    rdfs:label \"{class_name}\" ;\n"
            
            # Add description from docstring
            if class_info['docstring']:
                docstring = class_info['docstring'].strip().replace('\n', ' ').replace('"', '\\"')
                ttl += f"    rdfs:comment \"{docstring}\" ;\n"
            
            # Add parent classes (inheritance)
            if class_info['parents']:
                for parent in class_info['parents']:
                    ttl += f"    rdfs:subClassOf :{parent} ;\n"
            
            ttl += f"    dcterms:source \"{class_info['module']} module\" .\n\n"
        
        return ttl
    
    def _generate_properties(self) -> str:
        """Generate property definitions from dataclass fields"""
        ttl = "\n# ==================================================\n"
        ttl += "# PROPERTIES (from dataclass fields)\n"
        ttl += "# ==================================================\n\n"
        
        # Collect all unique properties across all classes
        all_properties = {}
        
        for class_name, class_info in self.dataclasses.items():
            for field_name, field_info in class_info['fields'].items():
                if field_name not in all_properties:
                    all_properties[field_name] = {
                        'type': field_info['type'],
                        'classes': []
                    }
                all_properties[field_name]['classes'].append(class_name)
        
        # Generate property definitions
        for prop_name, prop_info in sorted(all_properties.items()):
            ttl += f":{prop_name} a owl:DatatypeProperty ;\n"
            ttl += f"    rdfs:label \"{prop_name}\" ;\n"
            
            # Domain (classes that have this property)
            if len(prop_info['classes']) == 1:
                ttl += f"    rdfs:domain :{prop_info['classes'][0]} ;\n"
            else:
                ttl += f"    rdfs:domain [ a owl:Class ; owl:unionOf ( "
                ttl += " ".join([f":{c}" for c in prop_info['classes'][:5]])  # Limit to 5 for readability
                ttl += " ) ] ;\n"
            
            # Range (data type)
            xsd_type = self._python_type_to_xsd(prop_info['type'])
            ttl += f"    rdfs:range {xsd_type} .\n\n"
        
        return ttl
    
    def _generate_relationships(self) -> str:
        """Generate object properties for relationships"""
        ttl = "\n# ==================================================\n"
        ttl += "# RELATIONSHIPS (Object Properties)\n"
        ttl += "# ==================================================\n\n"
        
        # Generate unique relationships
        seen = set()
        for rel in self.relationships:
            key = (rel['source'], rel['target'], rel['property'])
            if key in seen:
                continue
            seen.add(key)
            
            ttl += f"# {rel['source']} -> {rel['target']}\n"
            ttl += f":{rel['property'].replace('_id', '')} a owl:ObjectProperty ;\n"
            ttl += f"    rdfs:label \"{rel['property']}\" ;\n"
            ttl += f"    rdfs:domain :{rel['source']} ;\n"
            ttl += f"    rdfs:range :{rel['target'].title()} ;\n"
            ttl += f"    rdfs:comment \"Reference from {rel['source']} to {rel['target']}\" .\n\n"
        
        # Add standard relationship types
        ttl += "\n# Standard Relationship Types\n"
        ttl += ":locatedIn a owl:ObjectProperty ;\n"
        ttl += "    rdfs:label \"located in\" ;\n"
        ttl += "    rdfs:comment \"Spatial containment relationship\" .\n\n"
        
        ttl += ":belongsTo a owl:ObjectProperty ;\n"
        ttl += "    rdfs:label \"belongs to\" ;\n"
        ttl += "    rdfs:comment \"Membership relationship\" .\n\n"
        
        ttl += ":connectedTo a owl:ObjectProperty ;\n"
        ttl += "    rdfs:label \"connected to\" ;\n"
        ttl += "    rdfs:comment \"Network connectivity relationship\" .\n\n"
        
        return ttl
    
    def _python_type_to_xsd(self, python_type: str) -> str:
        """Map Python types to XSD datatypes"""
        type_map = {
            'str': 'xsd:string',
            'int': 'xsd:integer',
            'float': 'xsd:float',
            'bool': 'xsd:boolean',
            'datetime': 'xsd:dateTime',
            'date': 'xsd:date',
            'time': 'xsd:time',
            'Any': 'xsd:anyType',
        }
        
        # Handle Optional types
        if 'Optional[' in python_type:
            inner = python_type.replace('Optional[', '').replace(']', '')
            return self._python_type_to_xsd(inner)
        
        # Handle List types
        if 'List[' in python_type:
            return 'rdf:List'
        
        # Handle Dict types
        if 'Dict[' in python_type:
            return 'xsd:string'  # Serialize as JSON string
        
        # Check direct mappings
        for key in type_map:
            if key in python_type:
                return type_map[key]
        
        # Default to string
        return 'xsd:string'


def main():
    """Main execution"""
    print("=" * 60)
    print("Transport Data Ontology - TTL Generator")
    print("=" * 60)
    
    extractor = OntologyExtractor()
    extractor.extract_all()
    
    output_file = ONTOLOGY_DIR / "transport_ontology.ttl"
    extractor.generate_ttl(str(output_file))
    
    print("\n" + "=" * 60)
    print("✅ TTL Generation Complete!")
    print("=" * 60)
    print(f"\nOutput file: {output_file}")
    print(f"File size: {output_file.stat().st_size:,} bytes")
    
    print("\n📚 To view/validate:")
    print("  - Protégé: https://protege.stanford.edu/")
    print("  - Online: http://www.ldf.fi/service/rdf-validator")
    print("  - Turtle Validator: https://www.w3.org/2015/03/ShExValidata/")


if __name__ == "__main__":
    main()

/*
License
*/

#ifndef MOTOMAN_DRIVER_MOTOMAN_MEMORY_H
#define MOTOMAN_DRIVER_MOTOMAN_MEMORY_H

namespace motoman
{
namespace yrc1000_memory
{
/**
 * \brief Enumeration of motoman M registers.
 */
class Mregister
{
public:
  /**
   * \brief Constructor
   */
  Mregister();

  /**
   * \brief Destructor
   */
  ~Mregister();

typedef enum
{
    RESERVE_,
    ANA_O_,
    ANA_I_,
    SYS_
} mregister;

struct reserve {


typedef enum
{
    POS_S = 1000010, //M010 = 2 octets inférieurs & M011 = 2 octets supérieurs
    POS_L = 1000012,
    POS_U = 1000014,
    POS_R = 1000016,
    POS_B = 1000020,
    POS_T = 1000022,

    POS_X = 1000030,
    POS_Y = 1000032,
    POS_Z = 1000034,
    POS_Rx = 1000036,
    POS_Ry = 1000040,
    POS_Rz = 1000042,
    POS_Re = 1000044

} POS; // Codé sur 4 octets

typedef enum
{
    VIT_S = 1000050,
    VIT_L = 1000052,
    VIT_U = 1000054,
    VIT_R = 1000056,
    VIT_B = 1000060,
    VIT_T = 1000062,

    VIT_TCP = 1000070,

} VIT; // Codé sur 4 octets

typedef enum
{
    TRQe_S = 1000310,
    TRQe_L = 1000311,
    TRQe_U = 1000312,
    TRQe_R = 1000313,
    TRQe_B = 1000314,
    TRQe_T = 1000315,

    TRQe_X = 1000323,
    TRQe_Y = 1000324,
    TRQe_Z = 1000325,

} TRQe; //e = Estimé

typedef enum
{
    Fe_X = 1000320,
    Fe_Y = 1000321,
    Fe_Z = 1000322,

    Fe_Totale = 1000326
  
} Fe; //e = Estimé


typedef enum
{
    SENSOR_CH1_S = 1000330,
    SENSOR_CH1_L = 1000331,
    SENSOR_CH1_U = 1000332,
    SENSOR_CH1_R = 1000333,
    SENSOR_CH1_B = 1000334,
    SENSOR_CH1_T = 1000335,
  
} SENSOR_CH1;

typedef enum
{
    SENSOR_CH2_S = 1000340,
    SENSOR_CH2_L = 1000341,
    SENSOR_CH2_U = 1000342,
    SENSOR_CH2_R = 1000343,
    SENSOR_CH2_B = 1000344,
    SENSOR_CH2_T = 1000345,
  
} SENSOR_CH2;

typedef enum
{
    TRQmax_S = 1000350,
    TRQmax_L = 1000351,
    TRQmax_U = 1000352,
    TRQmax_R = 1000353,
    TRQmax_B = 1000354,
    TRQmax_T = 1000355,

    TRQmax_X = 1000363,
    TRQmax_Y = 1000364,
    TRQmax_Z = 1000365
  
} TRQmax;

typedef enum
{
    Fmax_X = 1000360,
    Fmax_Y = 1000361,
    Fmax_Z = 1000362,
    
    Fmax_Totale = 1000366
  
} Fmax;
};
typedef struct reserve reserve;

struct ana_o {
    //Non défini
};
typedef struct ana_o ana_o;

struct ana_i {
    //Non défini
};
typedef struct ana_i ana_i;

struct sys {
    //Non défini
};
typedef struct sys sys;

};
//Mregister class

/**
 * \brief Enumeration of motoman State registers.
 */
class Status
{
public:

  /**
   * \brief Constructor
   */
  Status();

  /**
   * \brief Destructor
   */
  ~Status();

typedef enum
{
    REMOTE = 80011,
    PLAY   = 80012,
    TEACH  = 80013,
    HOLD   = 80015,
    START  = 80016,
    SRV_ON = 80017,
    SAFF   = 80023,
    STOP_URG   = 80025,
    STOP_TEACH = 80026,
    PORTE  = 80027,
    VITESSE_2 = 80040,
    VITESSE_1 = 80041,
    SRV_PRET  = 80053,
    SRV_STAT  = 80054,
    CHOC_HOLD = 80060,
    CHOC_URG  = 80065

} GLOBAL;

struct general_i {
    // Entrée-Sortie définie par l'utilisateur.
};
typedef struct general_i general_i;


struct general_o {
    // Entrée-Sortie définie par l'utilisateur.
};
typedef struct general_o general_o;


struct sfty_ms { 
typedef enum
{
    LIDAR_RED = 81385    //Signal MS-OUT 54

} SFTY_MS;  // Signaux de sécurité interne (MS-OUTXX).
};
typedef struct sfty_ms sfty_ms;


struct sfty_fs {
    // Non défini.
};
typedef struct sfty_fs sfty_fs;


};
// Status Class

} //yrc1000_memory
} //motoman

/* YRC1000

Extract taken from Motoplus/IoServer.h  
These addresses may change depending on the robot controller. So, refer also 
to the Yaskawa Motoman documentation on IO addressing and configuration.

In this case, we refered to the
"YRC1000 OPTIONS INSTRUCTIONS FOR Concurrent I/O (RE-CKI-A467)" documentation.

*/ 

#define GENERALINMIN (10)
#define GENERALINMAX (5127)

#define GENERALOUTMIN (10010)
#define GENERALOUTMAX (15127)

#define EXTERNALINMIN (20010)
#define EXTERNALINMAX (25127)

#define NETWORKINMIN (27010)
#define NETWORKINMAX (29567)

#define NETWORKOUTMIN (37010)
#define NETWORKOUTMAX (39567)

#define EXTERNALOUTMIN (30010)
#define EXTERNALOUTMAX (35127)

#define SPECIFICINMIN (40010)
#define SPECIFICINMAX (42567)

#define SPECIFICOUTMIN (50010)
#define SPECIFICOUTMAX (55127)

#define IFPANELMIN (60010)
#define IFPANELMAX (60647)

#define AUXRELAYMIN (70010)
#define AUXRELAYMAX (79997)

#define CONTROLSTATUSMIN (80010)
#define CONTROLSTATUSMAX (85127)

#define PSEUDOINPUTMIN (87010)
#define PSEUDOINPUTMAX (87207)

#define REGISTERMIN (1000000)
#define REGISTERMAX_READ (1000999)
#define REGISTERMAX_WRITE (1000559)

#endif  // MOTOMAN_MEMORY_H